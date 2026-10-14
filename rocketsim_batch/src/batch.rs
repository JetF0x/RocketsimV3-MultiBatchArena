use std::{ops::Range, sync::Mutex};

use rayon::{ThreadPool, ThreadPoolBuilder, prelude::*};
use rocketsim::{Arena, ArenaEvent, CarControls, Team};

use crate::{
    affinity::{physical_core_cpus, pin_current_thread},
    config::{BatchConfig, Scheduling, arena_seed},
    info::{ArenaInfo, StepInfo},
    layout::{ACTION_DIM, BALL_DIM, CAR_DIM, read_action, write_ball, write_car},
};

/// Many independent [`Arena`]s stepped together on a Rayon thread pool.
///
/// Inputs and outputs live in flat buffers (see [`crate::layout`]) so they
/// can be handed to numpy or a tensor library without per-arena work:
///
/// 1. Write actions with [`BatchArena::set_actions`] / [`BatchArena::actions_mut`].
/// 2. Call [`BatchArena::step`]: every arena applies its actions, simulates
///    `tick_skip` ticks, then writes its observations and [`StepInfo`].
/// 3. Read [`BatchArena::ball_obs`], [`BatchArena::car_obs`],
///    [`BatchArena::pad_obs`] and [`BatchArena::info`].
///
/// Each arena only touches its own state and its own buffer slices, so the
/// results are bit-identical for any thread count and [`Scheduling`] mode.
pub struct BatchArena {
    config: BatchConfig,
    pool: ThreadPool,
    pinned_cpus: Option<Vec<usize>>,
    arenas: Vec<Arena>,
    num_pads: usize,

    actions: Vec<f32>,
    ball_obs: Vec<f32>,
    car_obs: Vec<f32>,
    pad_obs: Vec<f32>,
    final_ball_obs: Vec<f32>,
    final_car_obs: Vec<f32>,
    info: StepInfo,
}

/// One arena's slices of every batch buffer.
struct Slot<'a> {
    actions: &'a [f32],
    ball: &'a mut [f32],
    cars: &'a mut [f32],
    pads: &'a mut [f32],
    final_ball: &'a mut [f32],
    final_cars: &'a mut [f32],
    info: ArenaInfo<'a>,
}

/// Splits `buf` into `n` chunks of `size`, allowing `size == 0`.
fn split<T>(buf: &mut [T], n: usize, size: usize) -> std::vec::IntoIter<&mut [T]> {
    debug_assert_eq!(buf.len(), n * size);
    let chunks: Vec<_> = if size == 0 {
        (0..n).map(|_| <&mut [T]>::default()).collect()
    } else {
        buf.chunks_mut(size).collect()
    };
    chunks.into_iter()
}

/// Arenas owned by thread `k` of `threads` under [`Scheduling::Static`].
fn block_range(k: usize, threads: usize, n: usize) -> Range<usize> {
    k * n / threads..(k + 1) * n / threads
}

/// Writes one arena's ball, car and boost pad observations.
fn write_obs(arena: &Arena, ball: &mut [f32], cars: &mut [f32], pads: &mut [f32]) {
    write_ball(ball, arena.get_ball_state());
    for (car_idx, row) in cars.chunks_exact_mut(CAR_DIM).enumerate() {
        let (info, state) = arena.get_car_info_and_state(car_idx);
        write_car(row, info, state);
    }
    for (pad_idx, val) in pads.iter_mut().enumerate() {
        *val = f32::from(u8::from(arena.get_boost_pad_state(pad_idx).is_active()));
    }
}

impl BatchArena {
    /// Builds `config.num_arenas` arenas, adds the cars, and resets every
    /// arena to a seeded random kickoff.
    ///
    /// # Panics
    ///
    /// Panics if the collision meshes for the game mode are not loaded (see
    /// [`crate::init`]; `TheVoid` needs none), if there are no arenas or cars,
    /// or if the thread pool cannot be created.
    #[must_use]
    pub fn new(config: BatchConfig) -> Self {
        assert!(config.num_arenas > 0, "BatchConfig::num_arenas must be > 0");
        assert!(
            config.cars_per_arena() > 0,
            "BatchConfig needs at least one car per arena"
        );

        let pinned_cpus = config.pin_threads.then(|| {
            config
                .pin_cpus
                .clone()
                .filter(|cpus| !cpus.is_empty())
                .unwrap_or_else(physical_core_cpus)
        });

        let mut builder = ThreadPoolBuilder::new().thread_name(|i| format!("rocketsim-batch-{i}"));
        // When pinning, default to one thread per pin target rather than one
        // per logical CPU, so two threads never share a core.
        let num_threads = config
            .num_threads
            .or_else(|| pinned_cpus.as_ref().map(Vec::len));
        if let Some(n) = num_threads {
            builder = builder.num_threads(n);
        }
        if let Some(cpus) = pinned_cpus.clone() {
            builder = builder.start_handler(move |i| pin_current_thread(cpus[i % cpus.len()]));
        }
        let pool = builder
            .build()
            .expect("failed to build rocketsim_batch thread pool");

        let make_arena = |i: usize| {
            let arena_config = config
                .arena
                .clone()
                .with_rng_seed(arena_seed(config.seed, i));
            let mut arena = Arena::new_with_config(arena_config);
            for _ in 0..config.blue_cars {
                arena.add_car(Team::Blue, config.car_body);
            }
            for _ in 0..config.orange_cars {
                arena.add_car(Team::Orange, config.car_body);
            }
            arena.reset_to_random_kickoff(None);
            arena
        };

        let n = config.num_arenas;
        let arenas: Vec<Arena> = match config.scheduling {
            Scheduling::WorkStealing => {
                pool.install(|| (0..n).into_par_iter().map(make_arena).collect())
            }
            // Each thread builds the arenas it will own, so their memory is
            // first touched (and cached) on that thread's core.
            Scheduling::Static => {
                let threads = pool.current_num_threads();
                pool.broadcast(|ctx| {
                    block_range(ctx.index(), threads, n)
                        .map(make_arena)
                        .collect::<Vec<_>>()
                })
                .into_iter()
                .flatten()
                .collect()
            }
        };

        let c = config.cars_per_arena();
        let num_pads = arenas[0].num_boost_pads();

        let mut batch = Self {
            pool,
            pinned_cpus,
            arenas,
            num_pads,
            actions: vec![0.0; n * c * ACTION_DIM],
            ball_obs: vec![0.0; n * BALL_DIM],
            car_obs: vec![0.0; n * c * CAR_DIM],
            pad_obs: vec![0.0; n * num_pads],
            final_ball_obs: vec![0.0; n * BALL_DIM],
            final_car_obs: vec![0.0; n * c * CAR_DIM],
            info: StepInfo::new(n, c),
            config,
        };
        batch.refresh_obs();
        batch
    }

    /// Runs `f` on every arena in parallel, with that arena's buffer slices.
    fn for_each_slot<F>(&mut self, f: F)
    where
        F: Fn(usize, &mut Arena, Slot<'_>) + Sync,
    {
        let n = self.arenas.len();
        let c = self.config.cars_per_arena();
        let info = &mut self.info;

        let mut actions = self.actions.chunks(c * ACTION_DIM);
        let mut ball = split(&mut self.ball_obs, n, BALL_DIM);
        let mut cars = split(&mut self.car_obs, n, c * CAR_DIM);
        let mut pads = split(&mut self.pad_obs, n, self.num_pads);
        let mut final_ball = split(&mut self.final_ball_obs, n, BALL_DIM);
        let mut final_cars = split(&mut self.final_car_obs, n, c * CAR_DIM);
        let mut goal = info.goal.iter_mut();
        let mut terminated = info.terminated.iter_mut();
        let mut truncated = info.truncated.iter_mut();
        let mut ticks_simulated = info.ticks_simulated.iter_mut();
        let mut episode_ticks = info.episode_ticks.iter_mut();
        let mut ball_touches = split(&mut info.ball_touches, n, c);
        let mut demos = split(&mut info.demos, n, c);
        let mut demoed = split(&mut info.demoed, n, c);
        let mut boost_pickups = split(&mut info.boost_pickups, n, c);

        let slots: Vec<Slot<'_>> = (0..n)
            .map(|_| Slot {
                actions: actions.next().unwrap(),
                ball: ball.next().unwrap(),
                cars: cars.next().unwrap(),
                pads: pads.next().unwrap(),
                final_ball: final_ball.next().unwrap(),
                final_cars: final_cars.next().unwrap(),
                info: ArenaInfo {
                    goal: goal.next().unwrap(),
                    terminated: terminated.next().unwrap(),
                    truncated: truncated.next().unwrap(),
                    ticks_simulated: ticks_simulated.next().unwrap(),
                    episode_ticks: episode_ticks.next().unwrap(),
                    ball_touches: ball_touches.next().unwrap(),
                    demos: demos.next().unwrap(),
                    demoed: demoed.next().unwrap(),
                    boost_pickups: boost_pickups.next().unwrap(),
                },
            })
            .collect();

        let arenas = &mut self.arenas;
        match self.config.scheduling {
            Scheduling::WorkStealing => self.pool.install(|| {
                arenas
                    .par_iter_mut()
                    .zip(slots)
                    .enumerate()
                    .for_each(|(i, (arena, slot))| f(i, arena, slot));
            }),
            Scheduling::Static => {
                // Hand thread `k` the same block of arenas every call. Each
                // mutex is locked once, by its own thread, so never contended.
                let threads = self.pool.current_num_threads();
                let mut items = arenas.iter_mut().zip(slots).enumerate();
                let blocks: Vec<Mutex<Option<Vec<_>>>> = (0..threads)
                    .map(|k| {
                        let len = block_range(k, threads, n).len();
                        Mutex::new(Some(items.by_ref().take(len).collect()))
                    })
                    .collect();
                self.pool.broadcast(|ctx| {
                    let block = blocks[ctx.index()]
                        .lock()
                        .expect("block mutex poisoned")
                        .take()
                        .expect("block already taken");
                    for (i, (arena, slot)) in block {
                        f(i, arena, slot);
                    }
                });
            }
        }
    }

    /// Steps every arena by the configured `tick_skip`.
    pub fn step(&mut self) {
        self.step_ticks(self.config.tick_skip);
    }

    /// Applies the current actions, then simulates up to `ticks` ticks in
    /// every arena in parallel. An arena stops early on the tick a goal is
    /// scored. Afterwards the observations and [`StepInfo`] are refreshed.
    ///
    /// With `auto_reset`, an arena whose episode ended is reset to a random
    /// kickoff before its observations are written; its last pre-reset
    /// observations go to [`BatchArena::final_ball_obs`] and
    /// [`BatchArena::final_car_obs`].
    pub fn step_ticks(&mut self, ticks: u32) {
        let auto_reset = self.config.auto_reset;
        let max_episode_ticks = self.config.max_episode_ticks;

        self.for_each_slot(|_, arena, slot| {
            let Slot {
                actions,
                ball,
                cars,
                pads,
                final_ball,
                final_cars,
                mut info,
            } = slot;

            // The previous step ended an episode and auto-reset the arena.
            if auto_reset && (*info.terminated | *info.truncated) != 0 {
                *info.episode_ticks = 0;
            }
            info.clear_step();

            for (car_idx, row) in actions.chunks_exact(ACTION_DIM).enumerate() {
                arena.set_car_controls(car_idx, read_action(row));
            }

            for _ in 0..ticks {
                for event in arena.step_tick() {
                    match *event {
                        ArenaEvent::CarHitBall(e) => info.ball_touches[e.car_idx] += 1,
                        ArenaEvent::CarHitCar(e) if e.is_demo => {
                            info.demos[e.bumper_car_idx] += 1;
                            info.demoed[e.victim_car_idx] += 1;
                        }
                        ArenaEvent::CarPickupBoost(e) => info.boost_pickups[e.car_idx] += 1,
                        _ => {}
                    }
                }
                *info.ticks_simulated += 1;
                *info.episode_ticks += 1;

                if arena.is_ball_scored() {
                    // The ball is in the goal (or through the floor) on the
                    // defending team's side, so the other team scored.
                    let ball_y = arena.get_ball_state().phys.pos.y;
                    *info.goal = Team::from_team_y(ball_y).opposite() as i8;
                    *info.terminated = 1;
                    break;
                }
            }

            if *info.terminated == 0
                && max_episode_ticks.is_some_and(|max| *info.episode_ticks >= max)
            {
                *info.truncated = 1;
            }

            if auto_reset && (*info.terminated | *info.truncated) != 0 {
                write_obs(arena, final_ball, final_cars, &mut []);
                arena.reset_to_random_kickoff(None);
            }

            write_obs(arena, ball, cars, pads);
        });
    }

    /// Resets arenas to a random kickoff, in parallel.
    ///
    /// `which` selects arena indices (`None` = all). With `seed`, arena `i`
    /// uses [`arena_seed`]`(seed, i)` for this reset only; otherwise each
    /// arena draws from its own seeded RNG. Episode tick counts and step
    /// flags of the reset arenas are cleared.
    ///
    /// # Panics
    ///
    /// Panics if an index in `which` is out of range.
    pub fn reset(&mut self, which: Option<&[usize]>, seed: Option<u64>) {
        let n = self.arenas.len();
        let mut mask = vec![which.is_none(); n];
        for &i in which.unwrap_or_default() {
            assert!(i < n, "arena index {i} out of range (num_arenas = {n})");
            mask[i] = true;
        }

        self.for_each_slot(|i, arena, mut slot| {
            if !mask[i] {
                return;
            }
            arena.reset_to_random_kickoff(seed.map(|s| arena_seed(s, i)));
            slot.info.clear_step();
            *slot.info.episode_ticks = 0;
            write_obs(arena, slot.ball, slot.cars, slot.pads);
        });
    }

    /// Runs `f(arena_idx, arena)` on every arena in parallel, then refreshes
    /// the observation buffers. Use this for custom state setting.
    pub fn par_for_each<F>(&mut self, f: F)
    where
        F: Fn(usize, &mut Arena) + Sync,
    {
        self.for_each_slot(|i, arena, slot| {
            f(i, arena);
            write_obs(arena, slot.ball, slot.cars, slot.pads);
        });
    }

    /// Rewrites every observation buffer from the arenas' current state.
    /// Call after editing arenas through [`BatchArena::arena_mut`].
    pub fn refresh_obs(&mut self) {
        self.for_each_slot(|_, arena, slot| write_obs(arena, slot.ball, slot.cars, slot.pads));
    }

    /// Rewrites one arena's observations from its current state. Cheaper
    /// than [`BatchArena::refresh_obs`] after editing a single arena.
    ///
    /// # Panics
    ///
    /// Panics if `idx` is out of range.
    pub fn refresh_arena_obs(&mut self, idx: usize) {
        let c = self.config.cars_per_arena();
        let p = self.num_pads;
        write_obs(
            &self.arenas[idx],
            &mut self.ball_obs[idx * BALL_DIM..(idx + 1) * BALL_DIM],
            &mut self.car_obs[idx * c * CAR_DIM..(idx + 1) * c * CAR_DIM],
            &mut self.pad_obs[idx * p..(idx + 1) * p],
        );
    }

    /// Sets every car's action from a flat `[N * C * ACTION_DIM]` buffer.
    ///
    /// # Panics
    ///
    /// Panics if `actions` has the wrong length.
    pub fn set_actions(&mut self, actions: &[f32]) {
        assert_eq!(
            actions.len(),
            self.actions.len(),
            "expected {} action values (num_arenas {} * cars {} * {ACTION_DIM}), got {}",
            self.actions.len(),
            self.num_arenas(),
            self.cars_per_arena(),
            actions.len()
        );
        self.actions.copy_from_slice(actions);
    }

    /// Sets every car's controls from `[N * C]` [`CarControls`].
    ///
    /// # Panics
    ///
    /// Panics if `controls` has the wrong length.
    pub fn set_controls(&mut self, controls: &[CarControls]) {
        let expected = self.num_arenas() * self.cars_per_arena();
        assert_eq!(
            controls.len(),
            expected,
            "expected {expected} CarControls (num_arenas * cars), got {}",
            controls.len()
        );
        for (row, ctrl) in self.actions.chunks_exact_mut(ACTION_DIM).zip(controls) {
            row.copy_from_slice(&ctrl.to_floats());
        }
    }

    /// The flat `[N * C * ACTION_DIM]` action buffer, for writing in place.
    pub fn actions_mut(&mut self) -> &mut [f32] {
        &mut self.actions
    }

    /// The flat `[N * C * ACTION_DIM]` action buffer.
    #[must_use]
    pub fn actions(&self) -> &[f32] {
        &self.actions
    }

    /// `[N * BALL_DIM]` ball observations.
    #[must_use]
    pub fn ball_obs(&self) -> &[f32] {
        &self.ball_obs
    }

    /// `[N * C * CAR_DIM]` car observations.
    #[must_use]
    pub fn car_obs(&self) -> &[f32] {
        &self.car_obs
    }

    /// `[N * P]` boost pad states (`1.0` = active).
    #[must_use]
    pub fn pad_obs(&self) -> &[f32] {
        &self.pad_obs
    }

    /// `[N * BALL_DIM]` last ball observation before an auto-reset. Only
    /// meaningful for arenas with `terminated | truncated` set this step.
    #[must_use]
    pub fn final_ball_obs(&self) -> &[f32] {
        &self.final_ball_obs
    }

    /// `[N * C * CAR_DIM]` last car observations before an auto-reset. Only
    /// meaningful for arenas with `terminated | truncated` set this step.
    #[must_use]
    pub fn final_car_obs(&self) -> &[f32] {
        &self.final_car_obs
    }

    /// Results of the last step.
    #[must_use]
    pub const fn info(&self) -> &StepInfo {
        &self.info
    }

    #[must_use]
    pub const fn config(&self) -> &BatchConfig {
        &self.config
    }

    #[must_use]
    pub fn num_arenas(&self) -> usize {
        self.arenas.len()
    }

    #[must_use]
    pub const fn cars_per_arena(&self) -> usize {
        self.config.cars_per_arena()
    }

    /// Boost pads per arena (`0` for `TheVoid`).
    #[must_use]
    pub const fn num_pads(&self) -> usize {
        self.num_pads
    }

    /// Logical CPUs the pool threads are pinned to (thread `i` uses entry
    /// `i % len`), or `None` when pinning is off.
    #[must_use]
    pub fn pinned_cpus(&self) -> Option<&[usize]> {
        self.pinned_cpus.as_deref()
    }

    /// Threads in the batch's pool.
    #[must_use]
    pub fn num_threads(&self) -> usize {
        self.pool.current_num_threads()
    }

    /// All arenas, for read-only access to the full RocketSim API.
    #[must_use]
    pub fn arenas(&self) -> &[Arena] {
        &self.arenas
    }

    /// One arena, for state setting with the full RocketSim API. Call
    /// [`BatchArena::refresh_obs`] afterwards to update the buffers.
    pub fn arena_mut(&mut self, idx: usize) -> &mut Arena {
        &mut self.arenas[idx]
    }
}
