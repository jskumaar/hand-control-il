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


def test_continuous_arm_joints_are_not_limited(env):
    """Joints 0/2/4/6 rotate freely; clamping them to +-pi breaks replay.

    The source sets jnt_range = [-pi, pi] on them to bound IK sampling, not as a
    physical stop. Enforcing it clamps joint 2 by ~0.17 rad on real demos, which
    is ~9 cm of palm error -- enough to miss the object entirely.
    """
    for j in (0, 2, 4, 6):
        assert not env.model.jnt_limited[j], f"joint {j} must be unlimited"
        assert not env._bounded[j]
    for j in (1, 3, 5):
        assert env.model.jnt_limited[j], f"joint {j} must keep its stop"


def test_step_does_not_clip_continuous_joints(env):
    """A target beyond pi on a continuous joint must survive to the controller."""
    import numpy as np
    from handil.envs.scene import home_qpos
    env.reset()
    tgt = home_qpos()
    tgt[2] = 3.7                      # real demos reach 3.7-4.1 rad here
    env.step(tgt)
    assert env._target[2] == pytest.approx(3.7)


def test_static_hold_of_an_off_home_posture_converges(env):
    """The arm must be able to hold a commanded posture to high precision.

    This is the test that caught the +-pi clamp: every joint converged to 1e-11
    except joint 2, which sat exactly at pi instead of its commanded 3.31.
    """
    import numpy as np
    from handil.envs.scene import home_qpos
    env.reset()
    tgt = home_qpos()
    tgt[0] += 0.15
    tgt[2] = 3.31                     # beyond pi on purpose
    tgt[4] += 0.8
    env.n_substeps = 25
    for _ in range(200):
        env.step(tgt)
    err = np.abs(env.data.qpos[:7] - tgt[:7])
    assert err.max() < 1e-6, f"per-joint error {err}"


def test_set_active_object_by_name_selects_that_body(env):
    """The bug this guards: obj_idx is in the TRACE's spawn frame, not the
    scene's body order, and they disagree on 3 of the 4 ICRA objects. Passing an
    index across that boundary activated the wrong object in replay_demos.py and
    test_pipeline_replay.py. Always address by name.
    """
    for name in env.scene.object_bodies:
        i = env.set_active_object(name)
        assert env.scene.object_bodies[i] == name


def test_trace_and_scene_object_frames_are_known_to_differ(env):
    """Documents the mismatch so nobody 'simplifies' the name lookup away."""
    if len(env.scene.object_bodies) < 4:
        pytest.skip("not the 4-object ICRA scene")
    # Spawn order recorded in the ICRA traces, measured from obj_qpos.
    trace_order = ["obj_036_wood_block", "obj_017_orange",
                   "obj_025_mug", "obj_009_gelatin_box"]
    if set(trace_order) != set(env.scene.object_bodies):
        pytest.skip("different object set")
    assert env.scene.object_bodies != trace_order, (
        "frames now agree; the name-based lookup is still correct but the "
        "warning comments can be revisited")


def test_env_and_dataset_observations_agree_on_the_v2_fields(env):
    """The contract that broke twice: dataset build_obs and env observe() must
    produce identical object conditioning for the same object.

    Guards both the obj_in_palm slicing bug and the trace-vs-scene index bug.
    """
    import json
    from pathlib import Path
    from handil.data.schema import obs_slice

    data = Path("data/v2")
    if not (data / "manifest.json").exists():
        pytest.skip("data/v2 not extracted")
    eps = json.loads((data / "manifest.json").read_text())["episodes"]

    checked = set()
    for e in eps:
        if e["obj"] in checked or e["obj"] not in env.scene.object_bodies:
            continue
        checked.add(e["obj"])
        ds_obs = np.load(data / f"{e['episode_id']}.npz", allow_pickle=True)["obs"]
        env.set_active_object(e["obj"])
        env_obs = env.reset()
        for field in ("obj_id_onehot", "obj_size", "obj_mass"):
            sl = obs_slice(field)
            np.testing.assert_allclose(
                env_obs[sl], ds_obs[0][sl], atol=1e-6,
                err_msg=f"{field} differs for {e['obj']}")
    assert len(checked) >= 2, "no comparable objects found"
    env.set_active_object(0)


def test_reset_jitter_spreads_the_start_pose(env):
    """Every demo starts from a bit-identical home pose; jitter exists to test
    whether a policy generalises off it, and for future varied collection."""
    import numpy as np
    rng = np.random.default_rng(0)
    poses = []
    for _ in range(8):
        env.reset(arm_jitter=0.05, rng=rng)
        poses.append(env.data.qpos[:7].copy())
    spread = np.array(poses).std(0)
    assert (spread > 0.005).all(), spread
    assert np.isfinite(env.data.qpos).all()


def test_reset_without_jitter_is_deterministic(env):
    """Zero jitter must reproduce the demonstrated start, so existing results
    stay comparable. (A ~1e-4 rad residual comes from the object-settling
    steps, which predate the jitter option.)"""
    import numpy as np
    from handil.envs.scene import home_qpos
    a = env.reset().copy()
    b = env.reset(arm_jitter=0.0).copy()
    np.testing.assert_allclose(a, b, atol=1e-3)
    np.testing.assert_allclose(env.data.qpos[:7], home_qpos()[:7], atol=1e-3)
