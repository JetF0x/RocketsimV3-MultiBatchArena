use rocketsim::{ArenaConfig, ArenaMemWeightMode, CarBodyConfig, GameMode};

/// How a [`crate::BatchArena`] spreads its arenas over the thread pool.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Scheduling {
    /// Thread `k` of `T` always steps the same contiguous block of arenas
    /// (`k*N/T .. (k+1)*N/T`), so each arena's state stays in one core's
    /// caches. Best for large batches (at least ~8 arenas per thread).
    #[default]
    Static,
    /// Rayon work stealing: idle threads take arenas from busy ones. Arenas
    /// move between cores from step to step, which costs cache misses, but
    /// uneven or tiny batches balance better.
    WorkStealing,
}

/// Settings for a [`crate::BatchArena`].
///
/// Every arena in a batch shares the same config and car layout: `blue_cars`
/// Blue cars first, then `orange_cars` Orange cars, so car `j` of every arena
/// is the same "slot".
#[derive(Clone, Debug)]
pub struct BatchConfig {
    /// Config used to build each arena. Its `rng_seed` is ignored: each arena
    /// is seeded from [`BatchConfig::seed`] and its index instead.
    pub arena: ArenaConfig,
    /// Number of arenas in the batch.
    pub num_arenas: usize,
    /// Blue cars per arena.
    pub blue_cars: usize,
    /// Orange cars per arena.
    pub orange_cars: usize,
    /// Body used for every car.
    pub car_body: CarBodyConfig,
    /// Worker threads. `None` uses one per logical core.
    pub num_threads: Option<usize>,
    /// Base seed. Arena `i` gets [`arena_seed`]`(seed, i)`.
    pub seed: u64,
    /// Ticks simulated per [`crate::BatchArena::step`] call when no explicit
    /// count is given (the usual RL "tick skip").
    pub tick_skip: u32,
    /// Reset an arena to a random kickoff as soon as its episode ends
    /// (goal or `max_episode_ticks`). The final pre-reset observation is
    /// kept in the batch's `final_*` buffers.
    pub auto_reset: bool,
    /// Truncate episodes after this many ticks.
    pub max_episode_ticks: Option<u64>,
    /// How arenas are assigned to threads.
    pub scheduling: Scheduling,
    /// Pin pool thread `i` to one logical CPU: `pin_cpus[i % len]`, or by
    /// default the first logical CPU of each physical core (see
    /// [`crate::physical_core_cpus`]). Best on a dedicated machine.
    pub pin_threads: bool,
    /// Logical CPUs to pin to when `pin_threads` is set. `None` picks one per
    /// physical core automatically.
    pub pin_cpus: Option<Vec<usize>>,
}

impl BatchConfig {
    /// A 1v1 batch of `num_arenas` arenas for `game_mode`, using the
    /// `Balanced` memory mode (~108 KiB per arena instead of ~915 KiB).
    #[must_use]
    pub fn new(game_mode: GameMode, num_arenas: usize) -> Self {
        Self {
            arena: ArenaConfig::new(game_mode).with_mem_weight_mode(ArenaMemWeightMode::Balanced),
            num_arenas,
            blue_cars: 1,
            orange_cars: 1,
            car_body: CarBodyConfig::OCTANE,
            num_threads: None,
            seed: 0,
            tick_skip: 8,
            auto_reset: false,
            max_episode_ticks: None,
            scheduling: Scheduling::Static,
            pin_threads: false,
            pin_cpus: None,
        }
    }

    #[must_use]
    pub fn with_arena_config(mut self, arena: ArenaConfig) -> Self {
        self.arena = arena;
        self
    }

    #[must_use]
    pub const fn with_teams(mut self, blue_cars: usize, orange_cars: usize) -> Self {
        self.blue_cars = blue_cars;
        self.orange_cars = orange_cars;
        self
    }

    #[must_use]
    pub const fn with_car_body(mut self, car_body: CarBodyConfig) -> Self {
        self.car_body = car_body;
        self
    }

    #[must_use]
    pub const fn with_num_threads(mut self, num_threads: Option<usize>) -> Self {
        self.num_threads = num_threads;
        self
    }

    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    #[must_use]
    pub const fn with_tick_skip(mut self, tick_skip: u32) -> Self {
        self.tick_skip = tick_skip;
        self
    }

    #[must_use]
    pub const fn with_auto_reset(mut self, auto_reset: bool) -> Self {
        self.auto_reset = auto_reset;
        self
    }

    #[must_use]
    pub const fn with_max_episode_ticks(mut self, max_episode_ticks: Option<u64>) -> Self {
        self.max_episode_ticks = max_episode_ticks;
        self
    }

    #[must_use]
    pub const fn with_scheduling(mut self, scheduling: Scheduling) -> Self {
        self.scheduling = scheduling;
        self
    }

    #[must_use]
    pub const fn with_pin_threads(mut self, pin_threads: bool) -> Self {
        self.pin_threads = pin_threads;
        self
    }

    /// Pin threads to these logical CPUs (also turns `pin_threads` on).
    #[must_use]
    pub fn with_pin_cpus(mut self, pin_cpus: Vec<usize>) -> Self {
        self.pin_cpus = Some(pin_cpus);
        self.pin_threads = true;
        self
    }

    /// Cars per arena.
    #[must_use]
    pub const fn cars_per_arena(&self) -> usize {
        self.blue_cars + self.orange_cars
    }
}

/// SplitMix64 finalizer, used to spread seeds across arenas.
#[must_use]
pub const fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// The seed arena `idx` of a batch with base seed `seed` is built with.
#[must_use]
pub const fn arena_seed(seed: u64, idx: usize) -> u64 {
    splitmix64(seed ^ splitmix64(idx as u64))
}
