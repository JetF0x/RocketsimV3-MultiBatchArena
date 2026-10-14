"""Python binding tests. Uses game_mode="void", which needs no collision meshes.

Set ROCKETSIM_COLLISION_MESHES to also run the Soccar test.
"""

import os
import threading
import time

import numpy as np
import pytest

import rocketsim_batch as rsb


def make(n=4, **kw):
    kw.setdefault("blue_cars", 2)
    kw.setdefault("orange_cars", 1)
    kw.setdefault("game_mode", "void")
    kw.setdefault("seed", 42)
    kw.setdefault("tick_skip", 4)
    return rsb.BatchArena(n, **kw)


def random_actions(env, rng):
    return rng.uniform(-1, 1, (env.num_arenas, env.cars_per_arena, rsb.ACTION_DIM)).astype(np.float32)


def test_shapes_and_dtypes():
    env = make(5)
    assert env.num_arenas == 5 and env.cars_per_arena == 3 and env.num_pads == 0
    assert env.ball_obs.shape == (5, rsb.BALL_DIM) and env.ball_obs.dtype == np.float32
    assert env.car_obs.shape == (5, 3, rsb.CAR_DIM) and env.car_obs.dtype == np.float32
    assert env.pad_obs.shape == (5, 0)
    assert env.goal.dtype == np.int8 and (env.goal == rsb.NO_GOAL).all()
    assert env.terminated.dtype == np.bool_
    assert env.ball_touches.shape == (5, 3) and env.ball_touches.dtype == np.uint32
    assert env.episode_ticks.dtype == np.uint64
    assert set(env.get_obs()) == {"ball", "cars", "pads"}
    assert {"goal", "terminated", "truncated", "final_ball"} <= set(env.get_info())
    np.testing.assert_array_equal(env.car_obs[:, :, rsb.CAR["team"]], [[0, 0, 1]] * 5)


def test_step_is_deterministic_and_thread_independent():
    rng = np.random.default_rng(0)
    actions = [random_actions(make(8), rng) for _ in range(20)]
    results = []
    for threads in (1, 4):
        env = make(8, num_threads=threads)
        assert env.num_threads == threads
        for a in actions:
            env.step(a)
        results.append((env.ball_obs, env.car_obs))
    np.testing.assert_array_equal(results[0][0], results[1][0])
    np.testing.assert_array_equal(results[0][1], results[1][1])
    assert (env.episode_ticks == 20 * 4).all()


def test_boost_accelerates_only_selected_car():
    env = make(2)
    before = env.car_obs
    actions = np.zeros((2, 3, rsb.ACTION_DIM), np.float32)
    actions[1, 0, rsb.ACTION["boost"]] = 1.0
    env.step(actions, ticks=30)
    after = env.car_obs
    assert after[1, 0, rsb.CAR["boost"]] < before[1, 0, rsb.CAR["boost"]]
    assert after[0, 0, rsb.CAR["boost"]] >= before[0, 0, rsb.CAR["boost"]]
    dv = after[1, 0, rsb.CAR["vel"]] - before[1, 0, rsb.CAR["vel"]]
    assert dv @ after[1, 0, rsb.CAR["forward"]] > 0
    assert (env.ticks_simulated == 30).all()


def test_actions_accept_flat_and_float64():
    env = make(2)
    env.step(np.zeros(2 * 3 * rsb.ACTION_DIM))  # float64, flat
    env.step([[[0.0] * 8] * 3] * 2)  # nested lists
    with pytest.raises(ValueError, match="action values"):
        env.step(np.zeros((2, 3, 7), np.float32))


def test_reset_subset_and_seeded():
    env = make(4)
    for _ in range(10):
        env.step()
    before = env.ball_obs
    env.reset([1, 3], seed=5)
    after = env.ball_obs
    np.testing.assert_array_equal(after[[0, 2]], before[[0, 2]])
    assert not np.array_equal(after[[1, 3]], before[[1, 3]])
    np.testing.assert_array_equal(env.episode_ticks, [40, 0, 40, 0])

    env.reset(seed=9)
    first = env.car_obs
    env.step()
    env.reset(seed=9)
    np.testing.assert_array_equal(first, env.car_obs)
    with pytest.raises(IndexError):
        env.reset([4])


def test_auto_reset_truncation():
    env = make(3, auto_reset=True, max_episode_ticks=10)
    kickoff_z = env.ball_obs[:, 2].copy()
    env.step()
    env.step()
    assert not env.truncated.any()
    env.step()
    assert env.truncated.all() and not env.terminated.any()
    np.testing.assert_array_equal(env.episode_ticks, [12, 12, 12])
    assert (env.final_ball_obs[:, 2] < kickoff_z).all()  # it was falling in the void
    np.testing.assert_array_equal(env.ball_obs[:, 2], kickoff_z)
    env.step()
    np.testing.assert_array_equal(env.episode_ticks, [4, 4, 4])


def test_state_setters():
    env = make(2)
    env.set_ball_state(1, (10.0, 20.0, 300.0), vel=(1.0, 2.0, 3.0))
    np.testing.assert_allclose(env.ball_obs[1, rsb.BALL["pos"]], [10, 20, 300])
    np.testing.assert_allclose(env.ball_obs[1, rsb.BALL["vel"]], [1, 2, 3])
    env.set_car_state(0, 2, pos=(5.0, 6.0, 700.0), boost=12.5)
    assert env.car_obs[0, 2, rsb.CAR["boost"]] == 12.5
    np.testing.assert_allclose(env.car_obs[0, 2, rsb.CAR["pos"]], [5, 6, 700])
    with pytest.raises(IndexError):
        env.set_car_state(0, 3, boost=1.0)


def test_bad_arguments():
    with pytest.raises(ValueError):
        rsb.BatchArena(2, game_mode="tennis")
    with pytest.raises(ValueError):
        rsb.BatchArena(2, game_mode="void", car_body="tank")
    with pytest.raises(ValueError):
        rsb.BatchArena(0, game_mode="void")
    if not rsb.is_initialized():
        with pytest.raises(RuntimeError, match="init"):
            rsb.BatchArena(2, game_mode="soccar")


def test_step_releases_gil():
    env = make(64, blue_cars=3, orange_cars=3)
    ticks = []
    stop = threading.Event()

    def counter():
        while not stop.is_set():
            ticks.append(1)
            time.sleep(0)

    t = threading.Thread(target=counter)
    t.start()
    try:
        for _ in range(5):
            env.step(ticks=120)
    finally:
        stop.set()
        t.join()
    assert len(ticks) > 5


@pytest.mark.skipif("ROCKETSIM_COLLISION_MESHES" not in os.environ, reason="needs collision meshes")
def test_soccar_goal():
    rsb.init(os.environ["ROCKETSIM_COLLISION_MESHES"])
    env = rsb.BatchArena(2, auto_reset=True)
    assert env.num_pads == 34 and env.pad_obs.all()
    pos, big = env.boost_pad_layout()
    assert pos.shape == (34, 3) and big.sum() == 6
    env.set_ball_state(0, (0, 4500, 100), vel=(0, 3000, 0))
    env.set_ball_state(1, (0, -4500, 100), vel=(0, -3000, 0))
    goals = np.full(2, rsb.NO_GOAL)
    for _ in range(60):
        env.step()
        goals = np.where(env.goal != rsb.NO_GOAL, env.goal, goals)
    np.testing.assert_array_equal(goals, [0, 1])
