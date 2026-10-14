//! Throughput of `BatchArena` at several thread counts vs. a plain
//! single-threaded loop over the same arenas.
//!
//! ```sh
//! cargo run --release -p rocketsim_batch --example bench_batch -- \
//!     --arenas 256 --threads 1,2,4 --ticks 8 --steps 500 [--meshes ./collision_meshes]
//! ```
//!
//! Without `--meshes` it runs `TheVoid` (no field), which needs no collision
//! meshes but has no world collisions. Pass `--meshes` to run Soccar.

use std::time::Instant;

use rocketsim_batch::{
    BatchArena, BatchConfig, arena_seed,
    layout::{self, ACTION_DIM},
    rocketsim::{Arena, GameMode, Team},
    splitmix64,
};

struct Args {
    arenas: usize,
    threads: Vec<usize>,
    ticks: u32,
    steps: u64,
    blue: usize,
    orange: usize,
    meshes: Option<String>,
}

fn parse_args() -> Args {
    let mut args = Args {
        arenas: 256,
        threads: vec![1, 2, 4],
        ticks: 8,
        steps: 500,
        blue: 1,
        orange: 1,
        meshes: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str() {
            "--arenas" => args.arenas = val().parse().unwrap(),
            "--threads" => {
                args.threads = val().split(',').map(|t| t.parse().unwrap()).collect();
            }
            "--ticks" => args.ticks = val().parse().unwrap(),
            "--steps" => args.steps = val().parse().unwrap(),
            "--blue" => args.blue = val().parse().unwrap(),
            "--orange" => args.orange = val().parse().unwrap(),
            "--meshes" => args.meshes = Some(val()),
            _ => panic!(
                "unknown flag {flag}; use --arenas --threads --ticks --steps --blue --orange --meshes"
            ),
        }
    }
    args
}

/// Cheap deterministic action stream, regenerated per step like a policy would.
fn fill_actions(buf: &mut [f32], step: u64) {
    for (i, v) in buf.iter_mut().enumerate() {
        let r = splitmix64(step.wrapping_mul(0x1_0000_0001) ^ i as u64);
        *v = (r >> 40) as f32 / (1u64 << 23) as f32 - 1.0;
    }
}

fn main() {
    let args = parse_args();
    let mode = if let Some(path) = &args.meshes {
        rocketsim_batch::init(path, true).expect("failed to load collision meshes");
        GameMode::Soccar
    } else {
        GameMode::TheVoid
    };
    let config = BatchConfig::new(mode, args.arenas)
        .with_teams(args.blue, args.orange)
        .with_tick_skip(args.ticks)
        .with_auto_reset(true)
        .with_max_episode_ticks(Some(120 * 30));
    let total_ticks = args.arenas as u64 * args.steps * u64::from(args.ticks);
    let c = config.cars_per_arena();
    println!(
        "{mode:?}, {} arenas x {}v{}, {} steps x {} ticks = {total_ticks} arena-ticks",
        args.arenas, args.blue, args.orange, args.steps, args.ticks
    );

    // Baseline: the same arenas stepped one after another on this thread.
    let mut arenas: Vec<Arena> = (0..args.arenas)
        .map(|i| {
            let mut arena =
                Arena::new_with_config(config.arena.clone().with_rng_seed(arena_seed(0, i)));
            (0..args.blue).for_each(|_| _ = arena.add_car(Team::Blue, config.car_body));
            (0..args.orange).for_each(|_| _ = arena.add_car(Team::Orange, config.car_body));
            arena.reset_to_random_kickoff(None);
            arena
        })
        .collect();
    let mut actions = vec![0.0; args.arenas * c * ACTION_DIM];
    let start = Instant::now();
    for s in 0..args.steps {
        fill_actions(&mut actions, s);
        for (arena, rows) in arenas.iter_mut().zip(actions.chunks(c * ACTION_DIM)) {
            for (car_idx, row) in rows.chunks(ACTION_DIM).enumerate() {
                arena.set_car_controls(car_idx, layout::read_action(row));
            }
            for _ in 0..args.ticks {
                arena.step_tick();
            }
            if arena.is_ball_scored() {
                arena.reset_to_random_kickoff(None);
            }
        }
    }
    let base = total_ticks as f64 / start.elapsed().as_secs_f64();
    println!("{:>24}: {:>12.0} ticks/s", "sequential loop", base);
    drop(arenas);

    for &threads in &args.threads {
        let mut batch = BatchArena::new(config.clone().with_num_threads(Some(threads)));
        let start = Instant::now();
        for s in 0..args.steps {
            fill_actions(batch.actions_mut(), s);
            batch.step();
        }
        let tps = total_ticks as f64 / start.elapsed().as_secs_f64();
        println!(
            "{:>24}: {:>12.0} ticks/s  ({:.2}x)",
            format!("BatchArena {threads} thread(s)"),
            tps,
            tps / base
        );
    }
}
