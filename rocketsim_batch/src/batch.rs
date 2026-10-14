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

/// Splits the first `n` items off `s` (any `n`, including 0).
fn take_mut<'a, T>(s: &mut &'a mut [T], n: usize) -> &'a mut [T] {
    let (head, tail) = std::mem::take(s).split_at_mut(n);
    *s = tail;
    head
}

/// Shared-slice version of [`take_mut`].
fn take_ref<'a, T>(s: &mut &'a [T], n: usize) -> &'a [T] {
    let (head, tail) = s.split_at(n);
    *s = tail;
    head
}

/// A contiguous run of arenas with their slices of every batch buffer.
///
/// Buffers are split per chunk on the calling thread (a handful of chunks per
/// pool thread), and each worker then carves per-arena [`Slot`]s out of its
/// own chunk. That keeps the serial work per step independent of the arena
/// count.
struct Chunk<'a> {
    start: usize,
    arenas: &'a mut [Arena],
    actions: &'a [f32],
    ball: &'a mut [f32],
    cars: &'a mut [f32],
    pads: &'a mut [f32],
    final_ball: &'a mut [f32],
    final_cars: &'a mut [f32],
    goal: &'a mut [i8],
    terminated: &'a mut [u8],
    truncated: &'a mut [u8],
    ticks_simulated: &'a mut [u32],
    episode_ticks: &'a mut [u64],
    ball_touches: &'a mut [u32],
    demos: &'a mut [u32],
    demoed: &'a mut [u32],
    boost_pickups: &'a mut [u32],
}

impl Chunk<'_> {
    /// Runs `f` on each arena of the chunk, in order.
    fn run<F>(mut self, cars: usize, pads: usize, f: &F)
    where
        F: Fn(usize, &mut Arena, Slot<'_>),
    {
        for (j, arena) in self.arenas.iter_mut().enumerate() {
            let slot = Slot {
                actions: take_ref(&mut self.actions, cars * ACTION_DIM),
                ball: take_mut(&mut self.ball, BALL_DIM),
                cars: take_mut(&mut self.cars, cars * CAR_DIM),
                pads: take_mut(&mut self.pads, pads),
                final_ball: take_mut(&mut self.final_ball, BALL_DIM),
                final_cars: take_mut(&mut self.final_cars, cars * CAR_DIM),
                info: ArenaInfo {
                    goal: &mut take_mut(&mut self.goal, 1)[0],
                    terminated: &mut take_mut(&mut self.terminated, 1)[0],
                    truncated: &mut take_mut(&mut self.truncated, 1)[0],
                    ticks_simulated: &mut take_mut(&mut self.ticks_simulated, 1)[0],
                    episode_ticks: &mut take_mut(&mut self.episode_ticks, 1)[0],
                    ball_touches: take_mut(&mut self.ball_touches, cars),
                    demos: take_mut(&mut self.demos, cars),
                    demoed: take_mut(&mut self.demoed, cars),
                    boost_pickups: take_mut(&mut self.boost_pickups, cars),
                },
            };
            f(self.start + j, arena, slot);
        }
    }
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
        let p = self.num_pads;
        let threads = self.pool.current_num_threads();

        // Static: one block per thread. Work stealing: ~8 chunks per thread,
        // enough for idle threads to balance load without per-arena work.
        let ranges: Vec<Range<usize>> = match self.config.scheduling {
            Scheduling::Static => (0..threads).map(|k| block_range(k, threads, n)).collect(),
            Scheduling::WorkStealing => {
                let size = n.div_ceil(threads * 8).max(1);
                (0..n).step_by(size).map(|s| s..(s + size).min(n)).collect()
            }
        };

        let info = &mut self.info;
        let mut arenas = self.arenas.as_mut_slice();
        let mut actions = self.actions.as_slice();
        let mut ball = self.ball_obs.as_mut_slice();
        let mut cars = self.car_obs.as_mut_slice();
        let mut pads = self.pad_obs.as_mut_slice();
        let mut final_ball = self.final_ball_obs.as_mut_slice();
        let mut final_cars = self.final_car_obs.as_mut_slice();
        let mut goal = info.goal.as_mut_slice();
        let mut terminated = info.terminated.as_mut_slice();
        let mut truncated = info.truncated.as_mut_slice();
        let mut ticks_simulated = info.ticks_simulated.as_mut_slice();
        let mut episode_ticks = info.episode_ticks.as_mut_slice();
        let mut ball_touches = info.ball_touches.as_mut_slice();
        let mut demos = info.demos.as_mut_slice();
        let mut demoed = info.demoed.as_mut_slice();
        let mut boost_pickups = info.boost_pickups.as_mut_slice();

        let chunks: Vec<Chunk<'_>> = ranges
            .into_iter()
            .map(|r| {
                let len = r.len();
                Chunk {
                    start: r.start,
                    arenas: take_mut(&mut arenas, len),
                    actions: take_ref(&mut actions, len * c * ACTION_DIM),
                    ball: take_mut(&mut ball, len * BALL_DIM),
                    cars: take_mut(&mut cars, len * c * CAR_DIM),
                    pads: take_mut(&mut pads, len * p),
                    final_ball: take_mut(&mut final_ball, len * BALL_DIM),
                    final_cars: take_mut(&mut final_cars, len * c * CAR_DIM),
                    goal: take_mut(&mut goal, len),
                    terminated: take_mut(&mut terminated, len),
                    truncated: take_mut(&mut truncated, len),
                    ticks_simulated: take_mut(&mut ticks_simulated, len),
                    episode_ticks: take_mut(&mut episode_ticks, len),
                    ball_touches: take_mut(&mut ball_touches, len * c),
                    demos: take_mut(&mut demos, len * c),
                    demoed: take_mut(&mut demoed, len * c),
                    boost_pickups: take_mut(&mut boost_pickups, len * c),
                }
            })
            .collect();

        match self.config.scheduling {
            Scheduling::WorkStealing => self.pool.install(|| {
                chunks.into_par_iter().for_each(|chunk| chunk.run(c, p, &f));
            }),
            Scheduling::Static => {
                // Thread `k` gets block `k` every call. Each mutex is locked
                // once, by its own thread, so it is never contended.
                let blocks: Vec<Mutex<Option<Chunk<'_>>>> =
                    chunks.into_iter().map(|ch| Mutex::new(Some(ch))).collect();
                self.pool.broadcast(|ctx| {
                    let chunk = blocks[ctx.index()]
                        .lock()
                        .expect("block mutex poisoned")
                        .take()
                        .expect("block already taken");
                    chunk.run(c, p, &f);
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
