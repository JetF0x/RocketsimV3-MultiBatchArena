from typing import Any, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

BALL_DIM: int
CAR_DIM: int
ACTION_DIM: int
NO_GOAL: int
TICK_RATE: float
BALL: dict[str, slice]
CAR: dict[str, Union[int, slice]]
ACTION: dict[str, int]

def init(collision_meshes_path: str = "./collision_meshes", silent: bool = True) -> None:
    """Load collision meshes (folder with soccar/, hoops/, dropshot/ .cmf files). Once per process."""

def is_initialized() -> bool: ...

class BatchArena:
    """N independent RocketSim v3 arenas with C cars each, stepped on a Rust thread pool.

    Cars are ordered blue first, then orange. Arena ``i`` is seeded from ``seed`` and ``i``,
    so results are reproducible and independent of ``num_threads``.
    """

    def __init__(
        self,
        num_arenas: int,
        blue_cars: int = 1,
        orange_cars: int = 1,
        game_mode: str = "soccar",
        car_body: str = "octane",
        num_threads: Optional[int] = None,
        seed: int = 0,
        mem_weight_mode: str = "balanced",
        tick_skip: int = 8,
        auto_reset: bool = False,
        max_episode_ticks: Optional[int] = None,
    ) -> None: ...
    def step(self, actions: Optional[npt.ArrayLike] = None, ticks: Optional[int] = None) -> None:
        """Apply (N, C, 8) actions (None = reuse last) and simulate ``ticks`` (default tick_skip).

        An arena stops early on the tick a goal is scored. With ``auto_reset`` an arena whose
        episode ended is reset to kickoff; its pre-reset state is in ``final_*_obs``.
        """
    def reset(self, indices: Optional[Sequence[int]] = None, seed: Optional[int] = None) -> None:
        """Reset the given arenas (default all) to a random kickoff."""
    def set_ball_state(
        self,
        arena: int,
        pos: Sequence[float],
        vel: Sequence[float] = (0.0, 0.0, 0.0),
        ang_vel: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> None: ...
    def set_car_state(
        self,
        arena: int,
        car: int,
        pos: Optional[Sequence[float]] = None,
        vel: Optional[Sequence[float]] = None,
        ang_vel: Optional[Sequence[float]] = None,
        boost: Optional[float] = None,
    ) -> None: ...
    def boost_pad_layout(self) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        """Pad positions (P, 3) and is-big flags (P,), in RLBot/RLGym order."""
    def get_obs(self) -> dict[str, npt.NDArray[np.float32]]: ...
    def get_info(self) -> dict[str, npt.NDArray[Any]]: ...
    @property
    def num_arenas(self) -> int: ...
    @property
    def cars_per_arena(self) -> int: ...
    @property
    def num_pads(self) -> int: ...
    @property
    def num_threads(self) -> int: ...
    @property
    def tick_skip(self) -> int: ...
    @property
    def ball_obs(self) -> npt.NDArray[np.float32]:
        """(N, BALL_DIM)"""
    @property
    def car_obs(self) -> npt.NDArray[np.float32]:
        """(N, C, CAR_DIM)"""
    @property
    def pad_obs(self) -> npt.NDArray[np.float32]:
        """(N, P), 1.0 = active"""
    @property
    def final_ball_obs(self) -> npt.NDArray[np.float32]: ...
    @property
    def final_car_obs(self) -> npt.NDArray[np.float32]: ...
    @property
    def actions(self) -> npt.NDArray[np.float32]: ...
    @property
    def goal(self) -> npt.NDArray[np.int8]:
        """(N,) scoring team this step (0 blue, 1 orange) or -1."""
    @property
    def terminated(self) -> npt.NDArray[np.bool_]: ...
    @property
    def truncated(self) -> npt.NDArray[np.bool_]: ...
    @property
    def ticks_simulated(self) -> npt.NDArray[np.uint32]: ...
    @property
    def episode_ticks(self) -> npt.NDArray[np.uint64]: ...
    @property
    def ball_touches(self) -> npt.NDArray[np.uint32]: ...
    @property
    def demos(self) -> npt.NDArray[np.uint32]: ...
    @property
    def demoed(self) -> npt.NDArray[np.uint32]: ...
    @property
    def boost_pickups(self) -> npt.NDArray[np.uint32]: ...
