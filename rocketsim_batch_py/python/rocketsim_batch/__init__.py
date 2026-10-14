"""Step many RocketSim v3 arenas in parallel from Python.

>>> import numpy as np, rocketsim_batch as rsb
>>> rsb.init("./collision_meshes")            # not needed for game_mode="void"
>>> env = rsb.BatchArena(1024, blue_cars=1, orange_cars=1, tick_skip=8, auto_reset=True)
>>> actions = np.zeros((env.num_arenas, env.cars_per_arena, rsb.ACTION_DIM), np.float32)
>>> env.step(actions)                          # GIL released while simulating
>>> env.car_obs.shape                          # (1024, 2, rsb.CAR_DIM)

Every array is a fresh copy. Units are RocketSim's: uu, uu/s, rad/s.
Use the index tables below (``CAR["boost"]`` etc.) to slice observations.
"""

from ._core import (
    ACTION_DIM,
    BALL_DIM,
    CAR_DIM,
    NO_GOAL,
    TICK_RATE,
    BatchArena,
    init,
    is_initialized,
)

#: Ball row layout: name -> slice into the last axis of ``ball_obs``.
BALL = {
    "pos": slice(0, 3),
    "vel": slice(3, 6),
    "ang_vel": slice(6, 9),
}

#: Car row layout: name -> index or slice into the last axis of ``car_obs``.
CAR = {
    "pos": slice(0, 3),
    "vel": slice(3, 6),
    "ang_vel": slice(6, 9),
    "forward": slice(9, 12),
    "up": slice(12, 15),
    "boost": 15,
    "on_ground": 16,
    "has_jumped": 17,
    "has_double_jumped": 18,
    "has_flipped": 19,
    "is_demoed": 20,
    "is_supersonic": 21,
    "team": 22,
    "demo_respawn_timer": 23,
}

#: Action row layout. Button slots (jump, boost, handbrake) are pressed when > 0.5.
ACTION = {
    "throttle": 0,
    "steer": 1,
    "pitch": 2,
    "yaw": 3,
    "roll": 4,
    "jump": 5,
    "boost": 6,
    "handbrake": 7,
}

__all__ = [
    "ACTION",
    "ACTION_DIM",
    "BALL",
    "BALL_DIM",
    "CAR",
    "CAR_DIM",
    "NO_GOAL",
    "TICK_RATE",
    "BatchArena",
    "init",
    "is_initialized",
]
