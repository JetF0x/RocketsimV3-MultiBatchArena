# RocketSim v3 — Reference

A compiled reference for **RocketSim v3**, the simulator this repo vendors in `vendor/rocketsim/`.
It was gathered from the upstream source at commit `6cf55d3` (2026-10-13). Where this file and
the code disagree, trust the code. Upstream's docs live on docs.rs (`rocketsim` crate) and in
`vendor/rocketsim/UPSTREAM_README.md`.

## 1. What it is

- RocketSim is a standalone simulation of Rocket League's gameplay logic and physics, originally written by
  **ZealanL** ([ZealanL/RocketSim](https://github.com/ZealanL/RocketSim)).
- **v3** is a ground-up **pure Rust** rewrite that includes its own port of the Bullet physics subset it needs.
  It lives on the upstream branch **`v3-rust`**. It is published on crates.io as **`rocketsim`**, version `0.2.11` at the time of
  vendoring, with 13 releases since 2026-09-25.
- License: **MIT** (© 2025 VirxEC). Edition 2024; the minimum Rust version is **1.95**.
- v2 (C++, branch `main`, v2.2.1) is still the default branch upstream. v2 was used from Python and Rust through
  third-party bindings (`mtheall/RocketSim` for Python, `VirxEC/rocketsim-rs` for Rust).

### v3 vs v2 (upstream claims)

| | v2 | v3 |
|---|---|---|
| Language | C++ plus Bullet | Pure Rust (no C++ toolchain) |
| Speed | baseline | **~4–5x faster** (random-bot and real-replay throughput) |
| Accuracy | good | more accurate on real replays, both open-loop segments and single-tick error |
| Memory per arena | n/a | tunable: 66 KiB to 915 KiB (`ArenaMemWeightMode`) |
| Determinism | seeded | seeded RNG, plus `Clone` that keeps the state exactly |

v3 is validated against a bundled 3v3 replay recording (`rocketsim_test/recordings/wisp_3v3_300s.rlpr.zst`)
using a segment-accuracy metric (`rlpr_metric`) and the throughput benchmarks `stress_v3` and `stress_v2`.

### Upstream workspace (`v3-rust` branch)

| Crate | Purpose | Vendored here? |
|---|---|---|
| `rocketsim` | The simulator library | **yes** (`vendor/rocketsim`) |
| `rocketsim_test` | Replay accuracy metric, stress benchmarks, v2 comparison (needs `rocketsim_rs` C++ bindings) | no |
| `rocketsim_vis` | Real-time 3D visualizer that implements the `Vis` trait | no (heavy graphics deps) |

## 2. Units, coordinates, timing

- **Tick rate: 120 Hz** (`consts::TICK_RATE`, `TICK_TIME = 1/120`). `Arena::step_tick` advances exactly one tick.
- **Unreal units (uu)**: 50 uu = 1 m (`consts::BT_TO_UU = 50`). Velocities are in uu/s and angular velocity in rad/s.
- `+Z` is up and `+X` is the car's forward axis. **Blue defends −Y; Orange defends +Y.**
- `PhysState.rot_mat` columns are forward (`x_axis`), right (`y_axis`) and up (`z_axis`).
- Gravity is −650 uu/s². Car max speed is 2300 uu/s, max angular speed 5.5 rad/s and supersonic speed 2200 uu/s.
- Boost runs from 0 to 100. A car spawns with 33.3 and uses 33.3 per second.

## 3. Collision meshes (required)

RocketSim ships **no game assets**. Before you create an arena for any mode other than `TheVoid`:

1. Dump the arena meshes from your own Rocket League install with
   [ZealanL/RLArenaCollisionDumper](https://github.com/ZealanL/RLArenaCollisionDumper).
2. Lay them out as `collision_meshes/{soccar,hoops,dropshot}/*.cmf`. Heatseeker and Snowday reuse the Soccar meshes.
3. Call one of the init functions once per process:
   - `rocketsim::init(path, silent)` loads from a folder.
   - `rocketsim::init_from_default(silent)` loads from `./collision_meshes/`.
   - `rocketsim::init_from_mem(map, silent)` loads from in-memory bytes.

   Later calls only log a warning. `is_initialized()` reports the state. Mesh files with unknown hashes are skipped with a warning.

`GameMode::TheVoid` needs no meshes. It has no field, goals or pads, and cars and the ball fall forever, so it suits unit
tests. This repo's tests use it.

## 4. Public API

### Arena lifecycle

```rust
rocketsim::init("./collision_meshes", true)?;
let mut arena = Arena::new_with_config(ArenaConfig::new(GameMode::Soccar).with_rng_seed(42));
let car = arena.add_car(Team::Blue, CarBodyConfig::OCTANE);   // returns the car index
arena.reset_to_random_kickoff(None);                            // None = use the arena RNG
arena.set_car_controls(car, CarControls::default().with_throttle(1.0));
for event in arena.step_tick() { /* ArenaEvent */ }
if arena.is_ball_scored() { arena.reset_to_random_kickoff(None); }
```

RocketSim does **not** keep score or reset itself after a goal. You check `is_ball_scored()` yourself.

### `Arena` methods

| Area | Methods |
|---|---|
| Create | `new(GameMode)`, `new_with_config(ArenaConfig)`, `add_car(Team, CarBodyConfig) -> usize` |
| Step | `step_tick() -> &[ArenaEvent]`, `get_last_step_events()`, `tick_count()` |
| Rules | `is_ball_scored()`, `reset_to_random_kickoff(Option<u64>)` (the seed overrides the RNG for that call only) |
| Ball | `get_ball_state() -> &BallState`, `set_ball_state(BallState)` |
| Cars | `num_cars()`, `cars()`, `get_car_info(i)`, `get_car_state(i)`, `get_car_info_and_state(i)`, `set_car_state(i, CarState)`, `get_car_controls(i)`, `set_car_controls(i, CarControls)`, `set_car_dodge_deadzone(i, f32)`, `respawn_car(i)`, `demolish_car(i)`, `reset_car_transient_contacts(i)`, `refresh_car_sticky_gate(i)` |
| Boost pads | `num_boost_pads()`, `get_boost_pad_state(i)`, `set_boost_pad_state(i, BoostPadState)`, `get_boost_pad_config(i)`, `get_all_boost_pad_states()`, `get_all_boost_pad_configs()` (pads are sorted in RLBot/RLGym order: by Y, then X) |
| Dropshot | `get_tile_states()`, `set_tile_states(TileStates)`, `num_tiles()` |
| Snapshot | `get_arena_state() -> ArenaState` (tick, cars, ball, pads, tiles; serde optional), `Clone` (full deterministic copy that shares the static meshes) |
| Queries | `cast_rays<const N>([RaycastQuery; N]) -> [Option<RaycastHitInfo>; N]` (fastest with multiples of 4), `test_collision(PhysState, Option<CarBodyConfig>, include_dynamics) -> bool` |
| Config | `get_config()`, `game_mode()`, `mutator_config()` |
| Internals | `clear_persistent_manifolds()`, `num_persistent_manifolds()` |
| Visualization | `vis: Option<Box<dyn Vis>>` (`Vis::update(&ArenaState, dt)` is called after each tick), `is_vis_enabled()` |
| RNG | `rng: fastrand::Rng` (public) |

### `ArenaConfig` (builder: `ArenaConfig::new(mode).with_*`)

| Field | Default | Notes |
|---|---|---|
| `game_mode` | Soccar | |
| `mutators` | `MutatorConfig::new(mode)` | see below |
| `mem_weight_mode` | `Heavy` | see §5 |
| `min_pos` / `max_pos` | (−5600, −6000, 0) / (5600, 6000, 2200) | broadphase bounds |
| `max_aabb_len` | 370 | broadphase cell size |
| `no_ball_rot` | false | ball never spins |
| `custom_boost_pads` | None | replaces the mode's layout |
| `rng_seed` | None | `Some` gives deterministic kickoffs and respawns |

### `GameMode`

`Soccar`, `Hoops`, `Heatseeker`, `Snowday`, `Dropshot`, `TheVoid`. `name()` gives the mesh folder names.

### `MutatorConfig` (per-mode defaults from `MutatorConfig::new`)

- **Physics:** `gravity`, `car_mass`, `ball_mass`, `ball_max_speed`, `ball_drag`, `ball_radius`, `ball_hit_extra_force_scale` and `bump_force_scale`.
- **Jumping:** `jump_accel` and `jump_immediate_force`.
- **Boost:**
  - `boost_accel_ground`, `boost_accel_air` and `boost_used_per_second`
  - `car_max_boost_amount` and `car_spawn_boost_amount`
  - `boost_pad_amount_{small,big}` and `boost_pad_cooldown_{small,big}`
  - `recharge_boost_enabled`, `recharge_boost_per_second` and `recharge_boost_delay`
- **Rules:** `respawn_delay`, `bump_cooldown_time`, `unlimited_flips`, `unlimited_double_jumps`, `goal_base_threshold_y` and `enable_team_demos`.
- **`demo_mode`:** `DemoMode::{Normal, OnContact, Disabled}`.

### Cars

- **`CarBodyConfig`** presets: `OCTANE`, `DOMINUS`, `PLANK`, `BREAKOUT`, `HYBRID`, `MERC` and `PSYCLOPS`.
- **`CarControls`** has `throttle`, `steer`, `pitch`, `yaw` and `roll` (each −1..1), plus `jump`, `boost` and `handbrake` (bools). Helpers:
  - `to_floats()` and `from_floats(arr, thresh)` work in that order, with 8 values (`CarControls::NUM_VALS`).
  - `clamp()`.
  - `with_*` builders.
- **`CarState`** fields:
  - `phys` (a `PhysState`), `controls` and `prev_controls`.
  - Ground contact: `is_on_ground` and `wheels_with_contact[4]`.
  - Jumping and flipping:
    - `has_jumped`, `has_double_jumped`, `has_flipped`, `is_jumping` and `jump_ticks`
    - `is_flipping`, `flip_time` and `flip_rel_torque`
    - `jump_hold_broken`, `air_time` and `air_time_since_jump`
  - Boost: `boost`, `is_boosting`, `boosting_time` and `time_since_boosted`.
  - Speed: `is_supersonic` and `supersonic_grace_timer`.
  - Handbrake and auto-flip: `handbrake_val`, `is_auto_flipping`, `auto_flip_timer` and `auto_flip_torque_scale`.
  - Bumps and contacts: `bump_cooldown_timer`, `bump_cooldown_victim`, `last_extra_hit_tick` and `world_contact_normal`.
  - Demos: `is_demoed` and `demo_respawn_timer`.
- **`CarInfo`** has `idx`, `team` and `config`.

### Ball

- **`BallState`** has `phys`, `hs_info` (Heatseeker: `y_target_dir`, `cur_target_speed`, `time_since_hit`), `ds_info` (Dropshot:
  `charge_level`, `accumulated_hit_force`, `y_target_dir`, `last_damage_tick`) and `tick_count_since_kickoff`.
- Use `BallState::DEFAULT` and `CarState::DEFAULT`; `PhysState` deliberately has no `Default`.

### Events (`ArenaEvent`, returned by every `step_tick`)

| Variant | Fields |
|---|---|
| `BallHitWorld` | `contact_point`, `contact_normal` |
| `CarHitBall` | `car_idx`, `contact_point`, `extra_hit_vel` |
| `CarHitCar` | `bumper_car_idx`, `victim_car_idx`, `contact_point`, `is_demo` |
| `CarHitWorld` | `car_idx`, `contact_point`, `contact_normal` |
| `CarPickupBoost` | `car_idx`, `boost_pad_idx` |
| `CarLanded` | `car_idx`, `wheels` |

### Features and logging

- Cargo feature `serde`: Serialize and Deserialize for the configs, states and events.
- Logging goes through the `log` and `fern` crates. `silent = true` on init suppresses the startup output.

## 5. Memory modes (`ArenaMemWeightMode`)

Upstream `stress_v3` numbers:

| Mode | 1v1 KiB per arena | 1v1 speed | 3v3 KiB per arena | 3v3 speed |
|---|---:|---|---:|---|
| `Heavy` (default) | ~915 | baseline | ~946 | baseline |
| `Balanced` | ~108 | ~1–2% slower | ~123 | ~3% slower |
| `Light` | ~66 | ~9% slower | ~91 | ~12% slower |

With thousands of arenas, `Heavy` costs about 1 GiB per 1000 arenas, so this repo's `BatchConfig::new` defaults to **`Balanced`**.

## 6. Determinism and threading

- With `rng_seed` set, an arena's kickoffs and respawns are reproducible. `Clone` copies all physics, contact history,
  pending events and the RNG state (`vis` becomes `None`). The clone and the original then step identically.
- `Arena` contains no `Rc` and no raw pointers. Static collision meshes live in process-global `RwLock`s and are
  shared read-only, so **`Arena: Send + Sync`**. Separate arenas can step on separate threads with no locking. That is the basis of `rocketsim_batch`.
- Upstream itself is single-threaded per arena; its stress benchmark loops arenas on one thread.

## 7. Upstream benchmarks and examples

```sh
cargo run --release --example stress_v3                 # random bots
cargo run --release --example stress_v3 -- --num-cars 0 # ball only
cargo run --release --example stress_v3 -- --rlpr-file rocketsim_test/recordings/wisp_3v3_300s.rlpr.zst
cargo run --features v2 --example rlpr_metric           # accuracy, v3 vs v2
```

The vendored crate keeps its examples (`basic_match`, `custom_mutators`, `game_modes`, `snapshot_raycast`). Run them with
`cargo run -p rocketsim --example basic_match`; they need `./collision_meshes`.

## 8. Usage and legal notice (from upstream)

- RocketSim contains no code from Rocket League. It is meant for ML environments and hobby projects that respect Rocket
  League's terms.
- Selling a playable clone is heavily discouraged. The author is strongly against using RocketSim-trained bots to cheat in
  online matches.

## 9. How this repo builds on it

`rocketsim_batch` (Rust) and the `rocketsim_batch` Python package (PyO3) wrap N arenas into one `BatchArena`. It steps
them on a Rayon thread pool and exchanges actions and observations through flat `f32` buffers. See the top-level
`README.md`.
