"""Tests for the demonstration-replay calibration path."""
from __future__ import annotations

import numpy as np
import pytest

from handil.data.schema import N_ROBOT, OBS_DIM

pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def env():
    from handil.envs.rollout import PickPlaceEnv
    try:
        return PickPlaceEnv()
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"hand-control scene unavailable: {e}")


def test_replay_of_a_static_action_stream_reports_no_pick(env):
    from handil.envs.scene import home_qpos
    from handil.eval import replay_episode

    actions = np.tile(home_qpos(), (40, 1))
    r = replay_episode(env, actions, episode_id="static", obj="none",
                       extra_steps=0)
    assert not r.success and not r.picked
    assert r.steps == 40
    assert r.n_drops == 0


def test_replay_restores_the_env_control_rate(env):
    """A per-episode substep schedule must not leak into later rollouts."""
    from handil.envs.scene import home_qpos
    from handil.eval import replay_episode

    before = env.n_substeps
    t = np.arange(10) * 0.05          # 20 Hz, unlike the env default
    replay_episode(env, np.tile(home_qpos(), (10, 1)), t=t, extra_steps=0)
    assert env.n_substeps == before


def test_replay_reports_arm_error_against_a_reference(env):
    from handil.envs.scene import home_qpos
    from handil.eval import replay_episode

    n = 20
    actions = np.tile(home_qpos(), (n, 1))
    obs_ref = np.zeros((n, OBS_DIM))
    obs_ref[:, :7] = home_qpos()[:7]          # the env should match this closely
    r = replay_episode(env, actions, obs_ref=obs_ref, extra_steps=0)
    assert np.isfinite(r.arm_err_p50)
    assert r.arm_err_p95 < 0.1, "holding home should track its own reference"


def test_replay_without_a_reference_leaves_error_nan(env):
    from handil.envs.scene import home_qpos
    from handil.eval import replay_episode

    r = replay_episode(env, np.tile(home_qpos(), (5, 1)), extra_steps=0)
    assert np.isnan(r.arm_err_p50)


def test_replay_tracks_the_active_object_not_object_zero(env):
    """min_obj_in_palm must measure the active object, not a stowed one.

    A stowed object sits ~6 m from the palm, so measuring the wrong one gives a
    distance no demonstration ever shows.
    """
    from handil.envs.scene import home_qpos
    from handil.eval import replay_episode

    last = len(env.scene.object_bodies) - 1
    if last == 0:
        pytest.skip("single-object scene")
    env.set_active_object(last)
    r = replay_episode(env, np.tile(home_qpos(), (10, 1)), extra_steps=0)
    env.set_active_object(0)
    assert r.min_obj_in_palm < 2.0, r.min_obj_in_palm
