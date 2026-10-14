//! Flat `f32` buffer layouts shared by the Rust and Python APIs.
//!
//! Every buffer is row-major with the arena index outermost, so arena `i`
//! owns one contiguous slice. All units are Unreal units (uu), uu/s and rad/s,
//! exactly as RocketSim reports them; nothing is normalized.

use rocketsim::{BallState, CarControls, CarInfo, CarState};

/// Values per ball row.
pub const BALL_DIM: usize = 9;

/// Ball row offsets.
pub mod ball {
    /// Position `[x, y, z]`.
    pub const POS: usize = 0;
    /// Linear velocity `[x, y, z]`.
    pub const VEL: usize = 3;
    /// Angular velocity `[x, y, z]`.
    pub const ANG_VEL: usize = 6;
}

/// Values per car row.
pub const CAR_DIM: usize = 24;

/// Car row offsets.
pub mod car {
    /// Position `[x, y, z]`.
    pub const POS: usize = 0;
    /// Linear velocity `[x, y, z]`.
    pub const VEL: usize = 3;
    /// Angular velocity `[x, y, z]`.
    pub const ANG_VEL: usize = 6;
    /// Rotation matrix forward column `[x, y, z]`.
    pub const FORWARD: usize = 9;
    /// Rotation matrix up column `[x, y, z]`.
    pub const UP: usize = 12;
    /// Boost amount, `0..=100`.
    pub const BOOST: usize = 15;
    /// `1.0` when on the ground.
    pub const ON_GROUND: usize = 16;
    /// `1.0` after the first jump.
    pub const HAS_JUMPED: usize = 17;
    /// `1.0` after a double jump.
    pub const HAS_DOUBLE_JUMPED: usize = 18;
    /// `1.0` after a flip (dodge).
    pub const HAS_FLIPPED: usize = 19;
    /// `1.0` while demolished.
    pub const IS_DEMOED: usize = 20;
    /// `1.0` while supersonic.
    pub const IS_SUPERSONIC: usize = 21;
    /// `0.0` for Blue, `1.0` for Orange.
    pub const TEAM: usize = 22;
    /// Seconds until a demolished car respawns.
    pub const DEMO_RESPAWN_TIMER: usize = 23;
}

/// Values per car action row.
pub const ACTION_DIM: usize = CarControls::NUM_VALS;

/// Action row offsets. Button slots count as pressed when `> 0.5`.
pub mod action {
    pub const THROTTLE: usize = 0;
    pub const STEER: usize = 1;
    pub const PITCH: usize = 2;
    pub const YAW: usize = 3;
    pub const ROLL: usize = 4;
    pub const JUMP: usize = 5;
    pub const BOOST: usize = 6;
    pub const HANDBRAKE: usize = 7;
}

/// Threshold above which an action button slot counts as pressed.
pub const BUTTON_THRESHOLD: f32 = 0.5;

#[inline]
fn put3(out: &mut [f32], at: usize, v: rocketsim::Vec3A) {
    out[at..at + 3].copy_from_slice(&v.to_array());
}

/// Writes one ball row (`out.len() == BALL_DIM`).
#[inline]
pub fn write_ball(out: &mut [f32], state: &BallState) {
    put3(out, ball::POS, state.phys.pos);
    put3(out, ball::VEL, state.phys.vel);
    put3(out, ball::ANG_VEL, state.phys.ang_vel);
}

/// Writes one car row (`out.len() == CAR_DIM`).
#[inline]
pub fn write_car(out: &mut [f32], info: &CarInfo, state: &CarState) {
    let phys = &state.phys;
    put3(out, car::POS, phys.pos);
    put3(out, car::VEL, phys.vel);
    put3(out, car::ANG_VEL, phys.ang_vel);
    put3(out, car::FORWARD, phys.rot_mat.x_axis);
    put3(out, car::UP, phys.rot_mat.z_axis);
    out[car::BOOST] = state.boost;
    out[car::ON_GROUND] = f32::from(u8::from(state.is_on_ground));
    out[car::HAS_JUMPED] = f32::from(u8::from(state.has_jumped));
    out[car::HAS_DOUBLE_JUMPED] = f32::from(u8::from(state.has_double_jumped));
    out[car::HAS_FLIPPED] = f32::from(u8::from(state.has_flipped));
    out[car::IS_DEMOED] = f32::from(u8::from(state.is_demoed));
    out[car::IS_SUPERSONIC] = f32::from(u8::from(state.is_supersonic));
    out[car::TEAM] = info.team as u8 as f32;
    out[car::DEMO_RESPAWN_TIMER] = state.demo_respawn_timer;
}

/// Decodes one action row (`row.len() == ACTION_DIM`).
#[inline]
pub fn read_action(row: &[f32]) -> CarControls {
    let mut vals = [0.0; ACTION_DIM];
    vals.copy_from_slice(row);
    CarControls::from_floats(vals, BUTTON_THRESHOLD).clamp()
}
