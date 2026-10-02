"""Tests for the internal-force squeeze controller.

These need the hand-control repo (the controller is imported from it, not
reimplemented) and are skipped without it.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("scipy")


@pytest.fixture(scope="module")
def scene():
    from handil.envs.scene import build_scene
    try:
        return build_scene("robocasa")
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"hand-control scene unavailable: {e}")


def test_grasp_controller_imports_without_casadi(scene):
    """grasp_control/__init__ pulls in casadi; we must not need it."""
    import sys
    from handil.envs.grasp import load_grasp_controller
    from handil.envs.scene import handcontrol_root
    GC = load_grasp_controller(handcontrol_root())
    assert hasattr(GC, "internal_force_torques")
    assert hasattr(GC, "slip_correction_torques")
    assert "casadi" not in sys.modules


def test_tip_sites_resolve_for_every_finger(scene):
    from handil.envs.grasp import FINGER_TIP_SITES, tip_site_ids
    sids = tip_site_ids(scene.model, tuple(FINGER_TIP_SITES))
    assert len(sids) == 4
    assert all(s >= 0 for s in sids)
    assert len(set(sids)) == 4


def test_live_contact_provider_normals_point_inward(scene):
    """Column 0 of each frame must be the INWARD surface normal."""
    import mujoco as mj
    from handil.envs.grasp import make_live_contact_provider, tip_site_ids
    from handil.envs.scene import home_qpos

    bid = mj.mj_name2id(scene.model, mj.mjtObj.mjOBJ_BODY,
                        scene.object_bodies[0])
    sids = tip_site_ids(scene.model, ("thumb", "index"))
    data = mj.MjData(scene.model)
    data.qpos[:23] = home_qpos()
    mj.mj_forward(scene.model, data)

    for p_tip, R in make_live_contact_provider(scene.model, bid, sids)(data):
        inward = R[:, 0]
        to_obj = data.xpos[bid] - p_tip
        to_obj /= np.linalg.norm(to_obj)
        np.testing.assert_allclose(inward, to_obj, atol=1e-9)
        # orthonormal, right-handed
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-9)


def test_squeeze_ramps_from_zero(scene):
    """Full force on a half-closed grasp shoves the object out; it must ramp."""
    import mujoco as mj
    from handil.envs.grasp import SqueezeController
    bid = mj.mj_name2id(scene.model, mj.mjtObj.mjOBJ_BODY,
                        scene.object_bodies[0])
    gids = [g for g in range(scene.model.ngeom)
            if scene.model.geom_bodyid[g] == bid]
    kp = np.concatenate([scene.kp_arm, np.full(16, 0.8)])
    kd = np.concatenate([np.full(7, 4.0), np.full(16, 0.05)])
    sq = SqueezeController(scene.model, bid, kp, kd, gamma=5.0,
                          obj_geom_ids=gids, dt=0.002, ramp_s=0.5)
    assert sq.ramp == 0.0
    data = mj.MjData(scene.model)
    mj.mj_forward(scene.model, data)
    for _ in range(10):
        sq.torques(data)
    assert 0.0 < sq.ramp < 1.0
    for _ in range(500):
        sq.torques(data)
    assert sq.ramp == pytest.approx(1.0)
    sq.reset()
    assert sq.ramp == 0.0


def test_squeeze_returns_finite_torque_without_contact(scene):
    """Before the fingers touch, the grasp map is degenerate -- must not blow up."""
    import mujoco as mj
    from handil.envs.grasp import SqueezeController
    from handil.envs.scene import home_qpos
    bid = mj.mj_name2id(scene.model, mj.mjtObj.mjOBJ_BODY,
                        scene.object_bodies[0])
    gids = [g for g in range(scene.model.ngeom)
            if scene.model.geom_bodyid[g] == bid]
    kp = np.concatenate([scene.kp_arm, np.full(16, 0.8)])
    kd = np.concatenate([np.full(7, 4.0), np.full(16, 0.05)])
    sq = SqueezeController(scene.model, bid, kp, kd, obj_geom_ids=gids)
    data = mj.MjData(scene.model)
    data.qpos[:23] = home_qpos()
    mj.mj_forward(scene.model, data)
    tau = sq.torques(data)
    assert tau.shape == (23,)
    assert np.isfinite(tau).all()


def test_env_with_squeeze_is_stable_holding_home(scene):
    """Adding the squeeze must not destabilise an otherwise-static hold."""
    from handil.envs.rollout import PickPlaceEnv
    from handil.envs.scene import home_qpos
    env = PickPlaceEnv(scene=scene, kp_hand=0.8, squeeze=True, gamma=5.0)
    assert env.squeeze is not None
    env.reset()
    home = home_qpos()
    for _ in range(100):
        env.step(home)
    assert np.isfinite(env.data.qpos).all()
    assert np.abs(env.data.qvel).max() < 10.0


def test_env_without_squeeze_has_no_controller(scene):
    from handil.envs.rollout import PickPlaceEnv
    env = PickPlaceEnv(scene=scene, kp_hand=2.0, squeeze=False)
    assert env.squeeze is None
