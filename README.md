# RocketSim v3 Multi-Batch Arena

Step **thousands of [RocketSim v3](https://github.com/ZealanL/RocketSim/tree/v3-rust) arenas at once**, from Rust or Python.

RocketSim v3 is the pure-Rust rewrite of RocketSim, the Rocket League physics simulator. Upstream reports it at roughly
4–5x the speed of v2. Upstream simulates one arena at a time. This repo adds a `BatchArena` that:

- owns N independent arenas, each with C cars, and steps them on a **Rayon thread pool**
- exchanges actions and observations through **flat `f32` buffers**, so a whole batch moves to or from numpy in one copy
- supports tick skip, goal detection, truncation, **auto-reset** (keeping the final observation) and per-car touch,
  demo and boost-pickup counters
- is **deterministic**: results are bit-identical for any thread count
- releases the Python **GIL** while stepping

Everything gathered about RocketSim v3 (API, units, meshes, memory modes, determinism, benchmarks) is in
**[docs/ROCKETSIM_V3.md](docs/ROCKETSIM_V3.md)**.

## Layout

| Path | What |
|---|---|
| `vendor/rocketsim/` | RocketSim v3 source, vendored unmodified from upstream `v3-rust` @ `6cf55d3` (see `vendor/UPSTREAM.md`) |
| `rocketsim_batch/` | Rust crate: `BatchArena`, `BatchConfig`, buffer `layout`, tests, benchmark |
| `rocketsim_batch_py/` | Python package `rocketsim_batch` (PyO3 + numpy, built with maturin) |
| `docs/ROCKETSIM_V3.md` | RocketSim v3 reference |

## Collision meshes

RocketSim needs Rocket League's arena collision meshes, which are **not** included. Dump them from your own game install with
[RLArenaCollisionDumper](https://github.com/ZealanL/RLArenaCollisionDumper) into `collision_meshes/{soccar,hoops,dropshot}/`.
Then call `init(path)` once per process. `game_mode = TheVoid` / `"void"` (no field) works without meshes.

## Rust

```rust
use rocketsim_batch::{BatchArena, BatchConfig, layout::{ACTION_DIM, CAR_DIM, car}, rocketsim::GameMode};

rocketsim_batch::init("./collision_meshes", true)?;
let mut batch = BatchArena::new(
    BatchConfig::new(GameMode::Soccar, 1024)   // 1024 arenas, Balanced memory mode
        .with_teams(1, 1)                       // blue cars first, then orange
        .with_tick_skip(8)
        .with_auto_reset(true)
        .with_max_episode_ticks(Some(120 * 300))
        .with_seed(0),
);

loop {
    policy(batch.car_obs(), batch.actions_mut());   // [N*C*CAR_DIM] in, [N*C*ACTION_DIM] out
    batch.step();                                    // all arenas, all threads
    let info = batch.info();                         // goal / terminated / truncated / touches / demos ...
}
```

To reach the full RocketSim API, use `batch.arenas()`, `batch.arena_mut(i)` followed by `refresh_arena_obs(i)`, or the
parallel `batch.par_for_each(|i, arena| ...)`.

## Python

```sh
pip install maturin
cd rocketsim_batch_py && maturin develop --release     # or: maturin build --release
```

```python
import numpy as np
import rocketsim_batch as rsb

rsb.init("./collision_meshes")
env = rsb.BatchArena(1024, blue_cars=1, orange_cars=1, tick_skip=8,
                     auto_reset=True, max_episode_ticks=120 * 300, seed=0)

actions = np.zeros((env.num_arenas, env.cars_per_arena, rsb.ACTION_DIM), np.float32)
env.step(actions)                        # GIL released while simulating

cars = env.car_obs                       # (N, C, CAR_DIM) float32
boost = cars[..., rsb.CAR["boost"]]      # (N, C)
done = env.terminated | env.truncated    # (N,) bool
scored_by = env.goal                     # (N,) int8: 0 blue, 1 orange, -1 none
last_ball = env.final_ball_obs[done]     # pre-reset state for bootstrapping
```

Other calls:
- `env.reset(indices=None, seed=None)`
- `env.set_ball_state(i, pos, vel, ang_vel)` and `env.set_car_state(i, car, pos=..., vel=..., boost=...)`
- `env.boost_pad_layout()`
- `env.get_obs()` and `env.get_info()`

Every property returns a fresh numpy copy. The constructor also takes `game_mode` (`soccar`, `hoops`, `heatseeker`,
`snowday`, `dropshot`, `void`), `car_body`, `num_threads` and `mem_weight_mode` (`heavy`, `balanced`, `light`).

## Buffer layout

All values are raw RocketSim units (uu, uu/s, rad/s). Arena index is outermost.

| Buffer | Shape | Row contents |
|---|---|---|
| ball obs | `(N, 9)` | pos 0:3, vel 3:6, ang_vel 6:9 |
| car obs | `(N, C, 24)` | see the table below |
| pad obs | `(N, P)` | 1.0 = active (P = 34 in Soccar, 0 in void) |
| actions | `(N, C, 8)` | throttle, steer, pitch, yaw, roll, jump, boost, handbrake (buttons pressed when > 0.5) |

Car row slots:

| Slots | Contents |
|---|---|
| 0:3 | pos |
| 3:6 | vel |
| 6:9 | ang_vel |
| 9:12 | forward |
| 12:15 | up |
| 15 | boost |
| 16 | on_ground |
| 17 | has_jumped |
| 18 | has_double_jumped |
| 19 | has_flipped |
| 20 | is_demoed |
| 21 | is_supersonic |
| 22 | team |
| 23 | demo_respawn_timer |

Per-step info arrays, all refreshed by each step:

| Array | Shape | Meaning |
|---|---|---|
| `goal` | `(N,)` | scoring team, or −1 |
| `terminated` | `(N,)` | episode ended by a goal |
| `truncated` | `(N,)` | episode hit `max_episode_ticks` |
| `ticks_simulated` | `(N,)` | ticks run this step (stepping stops at the goal tick) |
| `episode_ticks` | `(N,)` | ticks in the current episode |
| `ball_touches` | `(N, C)` | ball touches per car |
| `demos` | `(N, C)` | demos inflicted per car |
| `demoed` | `(N, C)` | times each car was demolished |
| `boost_pickups` | `(N, C)` | boost pads picked up per car |

## How stepping works

`step()` splits every buffer into per-arena slices and hands each arena and its slices to one Rayon task. Each task does
the following:

1. Apply the arena's actions.
2. Run `step_tick()` `tick_skip` times. Along the way it tallies events and checks `is_ball_scored()`, and stops early on a goal.
3. Flag truncation.
4. Optionally save the final observation and reset to a random kickoff.
5. Write the arena's observations into its own slices.

Without `auto_reset`, RocketSim leaves a scored ball in the net, so call `reset()` on those arenas yourself; otherwise
`goal` keeps firing on each later step.

Arenas share nothing mutable, so this needs no locks. Each arena is seeded from `(seed, index)`, so the results do not
depend on scheduling.

## Benchmarks

`cargo run --release -p rocketsim_batch --example bench_batch -- --arenas 256 --threads 1,2,4 --ticks 8 --steps 300 [--blue 3 --orange 3] [--meshes ./collision_meshes]`

These numbers come from a 4-vCPU cloud container in `TheVoid` mode (no field meshes were available). Soccar ticks cost more
per tick, which makes the parallel scaling better, not worse.

| Workload | Sequential loop | Batch, 1 thread | Batch, 2 threads | Batch, 4 threads |
|---|---:|---:|---:|---:|
| 256 arenas, 1v1 | 1.23 M ticks/s | 1.16 M (0.94x) | 1.87 M (1.52x) | 3.62 M (2.94x) |
| 256 arenas, 3v3 | 0.45 M ticks/s | 0.45 M (0.99x) | 0.89 M (1.98x) | 1.65 M (3.65x) |

From Python, 1024 arenas with 4 threads and numpy copies of the observations every step reached about 383k env-steps/s
for 1v1 and 183k for 3v3, at a tick skip of 8.

## Testing

```sh
cargo test --workspace --exclude rocketsim_batch_py        # Rust (TheVoid; no meshes needed)
cargo clippy --workspace --all-targets
cd rocketsim_batch_py && maturin develop --release && pytest tests
```

Set `ROCKETSIM_COLLISION_MESHES=/path/to/collision_meshes` to also run the Soccar goal, pad and auto-reset tests.

## License

The RocketSim v3 source in `vendor/rocketsim` is MIT licensed (© VirxEC); see `vendor/rocketsim/LICENSE`. The batch
layer is MIT as well. Please respect upstream's usage notice in `docs/ROCKETSIM_V3.md` §8.
