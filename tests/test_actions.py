"""Tests for action recovery -- the correctness-critical part of the repo."""
from __future__ import annotations

import numpy as np
import pytest

from handil.data import actions as act
from handil.data.schema import N_ARM, N_ROBOT

# Computed from models/scene_pick_place.xml at the wrist-down home pose; the
# source repo derives the same numbers at startup. Verified identical under
# MuJoCo 3.6 (the source repo's pin) and 3.14.
#
# Tolerance is 2% rather than tighter: joint 6 is the wrist roll, whose gain is
# ~3 against 509 at the shoulder, so the last digit moves with small inertia
# changes. A real regression -- a changed home pose or mount -- shifts these by
# far more than 2%.
KP_ARM_EXPECTED = np.array([509.3, 675.5, 254.3, 324.0, 65.9, 65.0, 3.05])
KP_ARM_RTOL = 2e-2


def test_inversion_is_exact_for_a_synthetic_pd_controller():
    """Round-trip: build torques from known targets, recover the targets."""
    rng = np.random.default_rng(0)
    n = 500
    qpos = rng.normal(size=(n, N_ROBOT))
    target = qpos + rng.normal(scale=0.1, size=(n, N_ROBOT))
    kp_arm, kp_hand = KP_ARM_EXPECTED, 2.0
    kp = np.concatenate([kp_arm, np.full(N_ROBOT - N_ARM, kp_hand)])

    bias = rng.normal(size=(n, N_ROBOT))          # gravity comp, added then removed
    qfrc = kp * (target - qpos) + bias

    got = act.targets_from_torque(qpos, qfrc, bias, kp_arm, kp_hand)
    np.testing.assert_allclose(got, target, atol=1e-9)


def test_kp_arm_matches_the_source_repo():
    """Guards the gains against a scene change silently shifting them."""
    mj = pytest.importorskip("mujoco")
    from handil.envs.scene import build_scene
    try:
        scene = build_scene("pick_place")
    except FileNotFoundError:
        pytest.skip("hand-control repo not available")
    np.testing.assert_allclose(scene.kp_arm, KP_ARM_EXPECTED, rtol=KP_ARM_RTOL)


def test_kp_hand_tracks_the_pipeline_not_the_retargeter():
    assert act.kp_hand_for_mode("contact_aware_teleop") == 0.8
    assert act.kp_hand_for_mode("contact_aware_w_anyteleop") == 0.8
    assert act.kp_hand_for_mode("dexpilot") == 2.0
    assert act.kp_hand_for_mode("anyteleop") == 2.0
    assert act.kp_hand_for_mode("something_new") == act.KP_HAND_DEFAULT


def test_next_qpos_fallback_shifts_by_one_and_holds_the_last():
    q = np.arange(12, dtype=float).reshape(4, 3)
    q = np.pad(q, ((0, 0), (0, N_ROBOT - 3)))
    nxt = act.targets_from_next_qpos(q)
    np.testing.assert_allclose(nxt[:-1], q[1:, :N_ROBOT])
    np.testing.assert_allclose(nxt[-1], q[-1, :N_ROBOT])


def test_grasp_repair_touches_fingers_only_and_only_in_grasp():
    n = 10
    qpos = np.zeros((n, N_ROBOT))
    qpos[:, :] = np.arange(n)[:, None]
    targets = np.full((n, N_ROBOT), 99.0)
    phase = np.array(["REACH"] * 5 + ["GRASP"] * 5)

    out, frac = act.blend_grasp_phase(targets, qpos, phase)
    assert frac == 0.5
    # REACH rows untouched everywhere
    np.testing.assert_allclose(out[:5], 99.0)
    # GRASP rows: arm untouched, fingers replaced by next qpos
    np.testing.assert_allclose(out[5:, :N_ARM], 99.0)
    np.testing.assert_allclose(out[5:, N_ARM:],
                               act.targets_from_next_qpos(qpos)[5:, N_ARM:])


def test_grasp_repair_is_a_noop_without_phase_data():
    targets = np.ones((4, N_ROBOT))
    out, frac = act.blend_grasp_phase(targets, np.zeros((4, N_ROBOT)), None)
    assert frac == 0.0
    np.testing.assert_allclose(out, targets)


def test_clip_to_limits_respects_unbounded_joints():
    jr = np.zeros((N_ROBOT, 2))
    jr[:, 0], jr[:, 1] = -1.0, 1.0
    jr[3] = [0.0, 0.0]                      # unbounded marker (hi == lo)
    targets = np.full((3, N_ROBOT), 5.0)
    out = act.clip_to_limits(targets, jr)
    assert out[:, 0].max() == 1.0
    assert out[:, 3].max() == 5.0           # untouched


def test_torques_from_log_is_the_commanded_torque():
    """The torque action space needs no inversion and no gains."""
    rng = np.random.default_rng(3)
    n = 50
    tau = rng.normal(size=(n, N_ROBOT))
    bias = rng.normal(size=(n, N_ROBOT))
    qfrc = tau + bias                       # what the controller writes
    got = act.torques_from_log(qfrc, bias)
    np.testing.assert_allclose(got, tau, atol=1e-12)


def test_torques_from_log_ignores_object_dofs():
    """Only the 23 robot DOFs are actions; freejoint columns are not."""
    n = 10
    qfrc = np.ones((n, 47))
    bias = np.zeros((n, 47))
    assert act.torques_from_log(qfrc, bias).shape == (n, N_ROBOT)
