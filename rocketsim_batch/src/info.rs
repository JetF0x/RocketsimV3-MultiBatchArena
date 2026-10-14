/// `goal` value for "no goal this step".
pub const NO_GOAL: i8 = -1;

/// Per-step results for every arena, refreshed by each
/// [`crate::BatchArena::step`].
///
/// `[N]` arrays are indexed by arena; `[N * C]` arrays by
/// `arena * cars_per_arena + car`.
#[derive(Clone, Debug, Default)]
pub struct StepInfo {
    /// `[N]` Team that scored this step (`0` Blue, `1` Orange) or [`NO_GOAL`].
    /// Stepping stops at the scoring tick.
    pub goal: Vec<i8>,
    /// `[N]` `1` when the episode ended this step because of a goal.
    pub terminated: Vec<u8>,
    /// `[N]` `1` when the episode hit `max_episode_ticks` this step.
    pub truncated: Vec<u8>,
    /// `[N]` Ticks simulated this step (less than requested after a goal).
    pub ticks_simulated: Vec<u32>,
    /// `[N]` Ticks since the episode began. With `auto_reset`, this is the
    /// length of the episode that just ended when `terminated | truncated`.
    pub episode_ticks: Vec<u64>,
    /// `[N * C]` Ball touches by each car this step.
    pub ball_touches: Vec<u32>,
    /// `[N * C]` Demos inflicted by each car this step.
    pub demos: Vec<u32>,
    /// `[N * C]` Times each car was demolished this step.
    pub demoed: Vec<u32>,
    /// `[N * C]` Boost pads picked up by each car this step.
    pub boost_pickups: Vec<u32>,
}

impl StepInfo {
    pub(crate) fn new(num_arenas: usize, cars_per_arena: usize) -> Self {
        let per_car = num_arenas * cars_per_arena;
        Self {
            goal: vec![NO_GOAL; num_arenas],
            terminated: vec![0; num_arenas],
            truncated: vec![0; num_arenas],
            ticks_simulated: vec![0; num_arenas],
            episode_ticks: vec![0; num_arenas],
            ball_touches: vec![0; per_car],
            demos: vec![0; per_car],
            demoed: vec![0; per_car],
            boost_pickups: vec![0; per_car],
        }
    }
}

/// One arena's mutable view into [`StepInfo`].
pub(crate) struct ArenaInfo<'a> {
    pub goal: &'a mut i8,
    pub terminated: &'a mut u8,
    pub truncated: &'a mut u8,
    pub ticks_simulated: &'a mut u32,
    pub episode_ticks: &'a mut u64,
    pub ball_touches: &'a mut [u32],
    pub demos: &'a mut [u32],
    pub demoed: &'a mut [u32],
    pub boost_pickups: &'a mut [u32],
}

impl ArenaInfo<'_> {
    pub fn clear_step(&mut self) {
        *self.goal = NO_GOAL;
        *self.terminated = 0;
        *self.truncated = 0;
        *self.ticks_simulated = 0;
        self.ball_touches.fill(0);
        self.demos.fill(0);
        self.demoed.fill(0);
        self.boost_pickups.fill(0);
    }
}
