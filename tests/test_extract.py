"""Tests for episode extraction, including the palm-frame reconstruction."""
from __future__ import annotations

import numpy as np
import pytest

from handil.data.extract import (_trace_object_index, _mat_to_quat,
                                 _object_in_palm, _quat_to_mat)


def test_quat_mat_round_trip():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(200, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 0] < 0] *= -1                  # canonical sign
    back = _mat_to_quat(_quat_to_mat(q))
    back[back[:, 0] < 0] *= -1
    np.testing.assert_allclose(back, q, atol=1e-8)


def test_quat_to_mat_is_a_rotation():
    rng = np.random.default_rng(1)
    q = rng.normal(size=(50, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    R = _quat_to_mat(q)
    eye = np.einsum("nij,nkj->nik", R, R)
    np.testing.assert_allclose(eye, np.broadcast_to(np.eye(3), eye.shape), atol=1e-9)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-9)


def test_mat_to_quat_handles_negative_trace():
    """The Shepperd branch for trace <= 0 (180-degree rotations)."""
    R = np.array([np.diag([1.0, -1.0, -1.0]),      # 180 about x
                  np.diag([-1.0, 1.0, -1.0]),      # 180 about y
                  np.diag([-1.0, -1.0, 1.0])])     # 180 about z
    q = _mat_to_quat(R)
    np.testing.assert_allclose(_quat_to_mat(q), R, atol=1e-8)


def test_object_in_palm_matches_a_hand_built_transform():
    n = 8
    trace = {
        "xpos": np.zeros((n, 4, 3)),
        "xquat": np.zeros((n, 4, 4)),
        "obj_qpos": np.zeros((n, 7)),
    }
    trace["xquat"][:, 1, 0] = 1.0                 # palm identity rotation
    trace["xpos"][:, 1, :] = [1.0, 2.0, 3.0]      # palm at (1,2,3)
    trace["obj_qpos"][:, :3] = [1.5, 2.0, 3.0]    # object 0.5 m ahead in x
    trace["obj_qpos"][:, 3] = 1.0

    p, q = _object_in_palm(trace, slice(0, n), palm_bid=1, obj_idx=0, n=n)
    np.testing.assert_allclose(p, np.broadcast_to([0.5, 0.0, 0.0], (n, 3)), atol=1e-12)
    np.testing.assert_allclose(np.abs(q[:, 0]), 1.0, atol=1e-12)   # no relative rot


def test_object_in_palm_returns_nan_without_the_needed_fields():
    n = 4
    p, q = _object_in_palm({}, slice(0, n), palm_bid=1, obj_idx=0, n=n)
    assert np.isnan(p).all() and np.isnan(q).all()


def test_active_object_is_the_live_one_not_the_stowed_ones():
    """Stowed objects free-fall well past their -5 park depth."""
    n = 100
    oq = np.zeros((n, 28))
    oq[:, 2] = -80.0        # obj0 stowed, long since fallen
    oq[:, 9] = 0.86         # obj1 live on the counter
    oq[:, 16] = -360.0      # obj2 stowed
    oq[:, 23] = -210.0      # obj3 stowed
    idx = _trace_object_index({"obj_qpos": oq}, slice(0, n), "obj_x", None)
    assert idx == 1


def test_active_object_falls_back_to_names_when_several_are_live():
    n = 10
    oq = np.zeros((n, 28))
    oq[:, 2] = oq[:, 9] = oq[:, 16] = oq[:, 23] = 0.86      # all live
    bodies = ["a", "b", "c", "d"]
    assert _trace_object_index({"obj_qpos": oq}, slice(0, n), "c", bodies) == 2


def test_active_object_defaults_to_zero_without_data():
    assert _trace_object_index({}, slice(0, 4), "whatever", None) == 0


@pytest.mark.slow
def test_palm_frame_reconstruction_matches_the_source_exactly():
    """Validate against rows where hand-control logged obj_in_palm itself.

    The source only fills that column for the proximate object while a trial is
    live, so compare only on rows where the object is above the floor. On a real
    ICRA run this agrees to ~1e-5 m over ~11k rows -- the residual is float32
    storage precision in the logged arrays, not a modelling difference.
    """
    import glob
    from pathlib import Path

    cands = sorted(glob.glob(
        "/home/aipexws6/sureshkj/hand-control/logs/ICRA_*/*/pose_trace.npz"))
    if not cands:
        pytest.skip("hand-control ICRA logs not available")

    from handil.envs.scene import build_scene
    try:
        scene = build_scene("robocasa")
    except (FileNotFoundError, KeyError):
        pytest.skip("hand-control scene unavailable")

    for path in cands:
        d = dict(np.load(path, allow_pickle=True))
        if "obj_in_palm" not in d or "xpos" not in d or "obj_qpos" not in d:
            continue
        gt = d["obj_in_palm"]
        finite = np.isfinite(gt).all(1)
        if finite.sum() < 200 or d["qpos"].shape[1] != scene.model.nq:
            continue
        n = len(gt)
        n_obj = d["obj_qpos"].shape[1] // 7
        errs = []
        for oi in range(n_obj):
            live = d["obj_qpos"][:, 7 * oi + 2] > -1.0
            m = finite & live
            if m.sum() < 50:
                continue
            got, _ = _object_in_palm(d, slice(0, n), scene.palm_bid, oi, n)
            errs.append(np.nanmedian(np.linalg.norm(got[m] - gt[m], axis=1)))
        if errs:
            # 0.1 mm: far below any physical scale here, far above float32 noise.
            assert min(errs) < 1e-4, f"{Path(path).parent.name}: {min(errs):.6f} m"
            return
    pytest.skip("no run had both logged obj_in_palm and the needed fields")


def test_phase_onehot_is_a_strict_superset():
    """An opt-in block must append, leaving every fixed field's slice valid."""
    from handil.data.extract import build_obs
    from handil.data.schema import OBS_DIM, obs_dim, obs_slice

    n = 12
    tr = {
        "qpos": np.zeros((n, 51)), "qvel": np.zeros((n, 47)),
        "obj_qpos": np.zeros((n, 28)), "xpos": np.zeros((n, 37, 3)),
        "xquat": np.tile([1.0, 0, 0, 0], (n, 37, 1)),
        "norm_force": np.zeros((n, 4)), "tan_force": np.zeros((n, 4)),
    }
    tr["obj_qpos"][:, 2] = 0.86
    tr["obj_qpos"][:, 3] = 1.0
    phase = np.array(["APPROACH"] * 4 + ["PICK"] * 4 + ["TRANSPORT"] * 4)

    base = build_obs(tr, slice(0, n), 9, obj_idx=0, obj_name="obj_025_mug")
    wide = build_obs(tr, slice(0, n), 9, obj_idx=0, obj_name="obj_025_mug",
                     phase=phase, extras=("phase_onehot",))

    assert base.shape == (n, OBS_DIM)
    assert wide.shape == (n, obs_dim(("phase_onehot",)))
    np.testing.assert_allclose(wide[:, :OBS_DIM], base, equal_nan=True)
    # the appended block is a valid one-hot
    blk = wide[:, OBS_DIM:]
    np.testing.assert_allclose(blk.sum(1), 1.0)
    assert blk[0].argmax() == 0 and blk[4].argmax() == 1 and blk[8].argmax() == 2


def test_phase_onehot_zero_fills_without_a_phase_array():
    from handil.data.extract import build_obs
    from handil.data.schema import OBS_DIM
    n = 5
    tr = {"qpos": np.zeros((n, 51)), "qvel": np.zeros((n, 47))}
    w = build_obs(tr, slice(0, n), None, obj_name="x", extras=("phase_onehot",))
    np.testing.assert_allclose(w[:, OBS_DIM:], 0.0)
