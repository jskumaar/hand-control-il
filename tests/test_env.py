"""Tests for the rollout environment.

These need the hand-control repo (for the scene XML and YCB meshes) and are
skipped without it.
"""
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


def test_scene_matches_the_source_configuration(env):
    m = env.model
    assert m.nu == N_ROBOT
    assert m.opt.timestep == pytest.approx(0.002)      # 500 Hz physics
    assert m.opt.impratio == pytest.approx(100.0)      # elliptic cone config
    assert env.scene.palm_bid >= 0
    assert env.scene.object_bodies, "no free-jointed objects in the scene"


def test_observation_shape_and_finiteness(env):
    obs = env.reset()
    assert obs.shape == (OBS_DIM,)
    assert np.isfinite(obs).all()


def test_holding_the_home_pose_is_stable(env):
    """The controller must not diverge when commanded to stand still.

    Catches a wrong-sign torque, a missing gravity-compensation term, or gains
    that no longer match the scene -- all of which blow up within a second.
    """
    from handil.envs.scene import home_qpos
    env.reset()
    home = home_qpos()
    for _ in range(100):
        env.step(home)
    assert np.isfinite(env.data.qpos).all()
    assert np.abs(env.data.qvel).max() < 5.0
    drift = np.abs(env.data.qpos[:N_ROBOT] - home).max()
    assert drift < 0.5, f"home pose drifted {drift:.3f} rad"


def test_step_clips_targets_into_joint_limits(env):
    env.reset()
    env.step(np.full(N_ROBOT, 1e3))
    lo = env.model.jnt_range[:N_ROBOT, 0]
    hi = env.model.jnt_range[:N_ROBOT, 1]
    bounded = hi > lo
    assert (env._target[bounded] <= hi[bounded] + 1e-9).all()
    assert (env._target[bounded] >= lo[bounded] - 1e-9).all()


def test_a_still_policy_neither_picks_nor_succeeds(env):
    """Guards the success criteria against trivially firing."""
    from handil.envs.rollout import rollout
    from handil.envs.scene import home_qpos
    home = home_qpos()
    r = rollout(env, lambda obs: home, max_steps=60)
    assert not r.success
    assert not r.picked
    assert r.outcome == "timeout"


def test_windowed_policy_feeds_the_requested_history():
    from handil.eval import WindowedPolicy

    class Fake:
        horizon = 4

        def reset(self):
            pass

        def act(self, win):
            self.seen = win.shape
            return np.zeros(N_ROBOT)

    f = Fake()
    wp = WindowedPolicy(f, f.horizon)
    wp.reset()
    for i in range(10):
        wp(np.full(OBS_DIM, i, dtype=float))
    assert f.seen == (4, OBS_DIM)     # capped at the horizon
