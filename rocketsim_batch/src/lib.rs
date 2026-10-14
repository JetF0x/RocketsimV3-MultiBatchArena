//! Step many [RocketSim v3](https://github.com/ZealanL/RocketSim/tree/v3-rust)
//! arenas in parallel.
//!
//! [`BatchArena`] owns `N` independent [`rocketsim::Arena`]s with `C` cars
//! each and steps them on a Rayon thread pool. Actions go in and
//! observations come out through flat `f32` buffers laid out as described in
//! [`layout`], so a whole batch can be moved to or from numpy in one copy.
//!
//! ```no_run
//! use rocketsim_batch::{BatchArena, BatchConfig, layout::ACTION_DIM, rocketsim::GameMode};
//!
//! rocketsim_batch::init("./collision_meshes", true).unwrap();
//! let mut batch = BatchArena::new(BatchConfig::new(GameMode::Soccar, 512).with_teams(1, 1));
//!
//! let actions = vec![0.0; batch.num_arenas() * batch.cars_per_arena() * ACTION_DIM];
//! for _ in 0..100 {
//!     batch.set_actions(&actions);
//!     batch.step(); // 8 ticks in every arena
//!     let _ball = batch.ball_obs();
//!     let _goals = &batch.info().goal;
//! }
//! ```

mod batch;
mod config;
mod info;
pub mod layout;

pub use batch::BatchArena;
pub use config::{BatchConfig, arena_seed, splitmix64};
pub use info::{NO_GOAL, StepInfo};
/// The vendored RocketSim v3 crate, for direct access to its full API.
pub use rocketsim;
pub use rocketsim::{init, init_from_default, is_initialized};
