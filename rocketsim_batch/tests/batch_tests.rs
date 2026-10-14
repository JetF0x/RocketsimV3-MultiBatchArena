//! `TheVoid` needs no collision meshes, so most tests run anywhere. Tests
//! that need a real field are skipped unless `ROCKETSIM_COLLISION_MESHES`
//! points at a dumped `collision_meshes` folder.

use rocketsim_batch::{
    BatchArena, BatchConfig, NO_GOAL, arena_seed,
    layout::{self, ACTION_DIM, BALL_DIM, CAR_DIM, action, car},
    rocketsim::{Arena, BallState, GameMode, Team, Vec3A},
    splitmix64,
};

fn void_config(num_arenas: usize) -> BatchConfig {
    BatchConfig::new(GameMode::TheVoid, num_arenas)
        .with_teams(2, 1)
        .with_seed(1234)
        .with_tick_skip(4)
}

/// Deterministic pseudo-random actions in `[-1, 1]`.
fn random_actions(len: usize, step: u64) -> Vec<f32> {
    (0..len as u64)
        .map(|i| {
            let r = splitmix64(step.wrapping_mul(0x1_0000_0001) ^ i);
            (r >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

fn run_batch(config: BatchConfig, steps: u64) -> BatchArena {
    let mut batch = BatchArena::new(config);
    let len = batch.actions().len();
    for s in 0..steps {
        batch.set_actions(&random_actions(len, s));
        batch.step();
    }
    batch
}

#[test]
fn matches_arenas_stepped_by_hand() {
    let config = void_config(5);
    let steps = 30;
    let batch = run_batch(config.clone(), steps);

    let c = config.cars_per_arena();
    let n = config.num_arenas;
    let mut arenas: Vec<Arena> = (0..n)
        .map(|i| {
            let mut arena = Arena::new_with_config(
                config
                    .arena
                    .clone()
                    .with_rng_seed(arena_seed(config.seed, i)),
            );
            for _ in 0..config.blue_cars {
                arena.add_car(Team::Blue, config.car_body);
            }
            for _ in 0..config.orange_cars {
                arena.add_car(Team::Orange, config.car_body);
            }
            arena.reset_to_random_kickoff(None);
            arena
        })
        .collect();

    for s in 0..steps {
        let actions = random_actions(n * c * ACTION_DIM, s);
        for (arena, rows) in arenas.iter_mut().zip(actions.chunks(c * ACTION_DIM)) {
            for (car_idx, row) in rows.chunks(ACTION_DIM).enumerate() {
                arena.set_car_controls(car_idx, layout::read_action(row));
            }
            for _ in 0..config.tick_skip {
                arena.step_tick();
            }
        }
    }

    let mut ball = vec![0.0; BALL_DIM];
    let mut car_row = vec![0.0; CAR_DIM];
    for (i, arena) in arenas.iter().enumerate() {
        layout::write_ball(&mut ball, arena.get_ball_state());
        assert_eq!(ball, batch.ball_obs()[i * BALL_DIM..(i + 1) * BALL_DIM]);
        for j in 0..c {
            let (info, state) = arena.get_car_info_and_state(j);
            layout::write_car(&mut car_row, info, state);
            let at = (i * c + j) * CAR_DIM;
            assert_eq!(
                car_row,
                batch.car_obs()[at..at + CAR_DIM],
                "arena {i} car {j}"
            );
        }
        assert_eq!(
            batch.info().episode_ticks[i],
            steps * u64::from(config.tick_skip)
        );
    }
}

#[test]
fn thread_count_does_not_change_results() {
    let one = run_batch(void_config(16).with_num_threads(Some(1)), 40);
    let four = run_batch(void_config(16).with_num_threads(Some(4)), 40);
    assert_eq!(one.num_threads(), 1);
    assert_eq!(four.num_threads(), 4);
    assert_eq!(one.ball_obs(), four.ball_obs());
    assert_eq!(one.car_obs(), four.car_obs());
}

#[test]
fn arenas_are_seeded_differently() {
    let batch = BatchArena::new(void_config(4));
    let c = batch.cars_per_arena();
    let first_car = |i: usize| &batch.car_obs()[i * c * CAR_DIM..][..3];
    // TheVoid uses Soccar kickoff spots, so with 4+ arenas at least two
    // should pick different kickoffs.
    assert!((1..4).any(|i| first_car(0) != first_car(i)));
}

#[test]
fn obs_layout_matches_car_state() {
    let mut config = void_config(2);
    config.num_threads = Some(2);
    let mut batch = BatchArena::new(config);
    let c = batch.cars_per_arena();

    // Boost car 0 of arena 1 only.
    let mut actions = vec![0.0; batch.actions().len()];
    actions[c * ACTION_DIM + action::BOOST] = 1.0;
    batch.set_actions(&actions);

    let before = batch.car_obs().to_vec();
    batch.step_ticks(30);
    let after = batch.car_obs();

    let row = |obs: &[f32], arena: usize, car_idx: usize| {
        obs[(arena * c + car_idx) * CAR_DIM..][..CAR_DIM].to_vec()
    };
    let a1_before = row(&before, 1, 0);
    let a1_after = row(after, 1, 0);
    let a0_after = row(after, 0, 0);

    // Boost was spent and the velocity gained points along the car's forward axis.
    assert!(a1_after[car::BOOST] < a1_before[car::BOOST]);
    assert!(a0_after[car::BOOST] >= row(&before, 0, 0)[car::BOOST]);
    let dv = Vec3A::from_slice(&a1_after[car::VEL..]) - Vec3A::from_slice(&a1_before[car::VEL..]);
    let fwd = Vec3A::from_slice(&a1_after[car::FORWARD..]);
    assert!(
        dv.dot(fwd) > 0.0,
        "boost should accelerate forward, dv={dv}"
    );

    // Team column: blue, blue, orange.
    assert_eq!(row(after, 0, 0)[car::TEAM], 0.0);
    assert_eq!(row(after, 0, 1)[car::TEAM], 0.0);
    assert_eq!(row(after, 0, 2)[car::TEAM], 1.0);

    // Buffers agree with the arenas.
    let state = batch.arenas()[1].get_car_state(0);
    assert_eq!(a1_after[car::POS..car::POS + 3], state.phys.pos.to_array());
    assert_eq!(a1_after[car::BOOST], state.boost);
    assert_eq!(batch.info().ticks_simulated, vec![30, 30]);
}

#[test]
fn reset_only_touches_selected_arenas() {
    let mut batch = run_batch(void_config(6), 20);
    let before = batch.ball_obs().to_vec();

    batch.reset(Some(&[1, 4]), Some(99));
    for i in 0..6 {
        let row = &batch.ball_obs()[i * BALL_DIM..][..BALL_DIM];
        let old = &before[i * BALL_DIM..][..BALL_DIM];
        if i == 1 || i == 4 {
            assert_ne!(row, old);
            assert_eq!(batch.info().episode_ticks[i], 0);
        } else {
            assert_eq!(row, old);
            assert_eq!(batch.info().episode_ticks[i], 80);
        }
    }

    // A seeded reset is reproducible.
    let snapshot = batch.car_obs().to_vec();
    batch.reset(None, Some(7));
    let first = batch.car_obs().to_vec();
    batch.step();
    batch.reset(None, Some(7));
    assert_eq!(first, batch.car_obs());
    assert_ne!(snapshot, first);
}

#[test]
fn truncation_and_auto_reset() {
    let config = void_config(3)
        .with_auto_reset(true)
        .with_max_episode_ticks(Some(10));
    let mut batch = BatchArena::new(config);
    let kickoff_ball = batch.ball_obs().to_vec();

    batch.step(); // 4 ticks
    batch.step(); // 8 ticks
    assert_eq!(batch.info().truncated, vec![0, 0, 0]);
    let falling_ball = batch.ball_obs().to_vec();

    batch.step(); // 12 ticks >= 10: truncated, then reset
    assert_eq!(batch.info().truncated, vec![1, 1, 1]);
    assert_eq!(batch.info().terminated, vec![0, 0, 0]);
    assert_eq!(batch.info().goal, vec![NO_GOAL; 3]);
    assert_eq!(batch.info().episode_ticks, vec![12, 12, 12]);
    // In TheVoid the ball falls, so the pre-reset ball is below the falling one.
    for i in 0..3 {
        assert!(batch.final_ball_obs()[i * BALL_DIM + 2] < falling_ball[i * BALL_DIM + 2]);
    }
    assert_eq!(
        batch.ball_obs()[2],
        kickoff_ball[2],
        "ball back at kickoff height"
    );

    batch.step();
    assert_eq!(batch.info().truncated, vec![0, 0, 0]);
    assert_eq!(batch.info().episode_ticks, vec![4, 4, 4]);
}

#[test]
fn par_for_each_refreshes_obs() {
    let mut batch = BatchArena::new(void_config(3));
    batch.par_for_each(|i, arena| {
        let mut ball = BallState::DEFAULT;
        ball.phys.pos = Vec3A::new(i as f32 * 100.0, 0.0, 500.0);
        arena.set_ball_state(ball);
    });
    for i in 0..3 {
        assert_eq!(batch.ball_obs()[i * BALL_DIM], i as f32 * 100.0);
    }

    let mut ball = BallState::DEFAULT;
    ball.phys.pos = Vec3A::new(-7.0, 0.0, 500.0);
    batch.arena_mut(1).set_ball_state(ball);
    batch.refresh_arena_obs(1);
    assert_eq!(batch.ball_obs()[BALL_DIM], -7.0);
    assert_eq!(batch.ball_obs()[2 * BALL_DIM], 200.0);
}

#[test]
#[should_panic(expected = "action values")]
fn wrong_action_length_panics() {
    let mut batch = BatchArena::new(void_config(2));
    batch.set_actions(&[0.0; 3]);
}

/// Loads real collision meshes if `ROCKETSIM_COLLISION_MESHES` is set.
fn meshes_available() -> bool {
    match std::env::var("ROCKETSIM_COLLISION_MESHES") {
        Ok(path) => {
            rocketsim_batch::init(path, true).expect("failed to load collision meshes");
            true
        }
        Err(_) => {
            eprintln!("ROCKETSIM_COLLISION_MESHES not set; skipping Soccar test");
            false
        }
    }
}

#[test]
fn soccar_goal_detection_and_pads() {
    if !meshes_available() {
        return;
    }
    let config = BatchConfig::new(GameMode::Soccar, 2)
        .with_teams(1, 1)
        .with_auto_reset(true);
    let mut batch = BatchArena::new(config);
    assert_eq!(batch.num_pads(), 34);
    assert!(batch.pad_obs().iter().all(|&p| p == 1.0));

    // Fire the ball into the orange goal (+Y) in arena 0 and the blue goal in arena 1.
    for (i, dir) in [(0, 1.0), (1, -1.0)] {
        let mut ball = BallState::DEFAULT;
        ball.phys.pos = Vec3A::new(0.0, dir * 4500.0, 100.0);
        ball.phys.vel = Vec3A::new(0.0, dir * 3000.0, 0.0);
        batch.arena_mut(i).set_ball_state(ball);
    }
    batch.refresh_obs();

    let mut goals = [NO_GOAL; 2];
    for _ in 0..60 {
        batch.step();
        for (i, goal) in goals.iter_mut().enumerate() {
            if batch.info().goal[i] != NO_GOAL {
                *goal = batch.info().goal[i];
                assert_eq!(batch.info().terminated[i], 1);
                assert!(batch.final_ball_obs()[i * BALL_DIM + 1].abs() > 5000.0);
                assert!(
                    batch.ball_obs()[i * BALL_DIM + 1].abs() < 1.0,
                    "auto-reset to kickoff"
                );
            }
        }
    }
    assert_eq!(goals, [Team::Blue as i8, Team::Orange as i8]);
}

#[test]
fn arena_and_batch_are_thread_safe() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Arena>();
    assert_send_sync::<BatchArena>();
}
