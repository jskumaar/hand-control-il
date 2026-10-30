"""Canonical observation/action schema for hand-control demonstrations.

The source logs are not schema-stable: 8 distinct key-sets exist across
logs/**/pose_trace.npz, with qpos widths of 30 / 51 / 58 (1, 4 and 5 object
scenes) and norm_force widths of 2 or 4 (pre/post the fixed-width fix at
kinova_leap_pick_place.py:6041). Everything here is defined against the
*robot* slice, which is invariant at 23 DOF, plus a single active object.
"""
from __future__ import annotations

import numpy as np

N_ARM = 7
N_HAND = 16
N_ROBOT = N_ARM + N_HAND  # 23

FINGERS = ("thumb", "index", "middle", "ring")

# On-disk contract version. Bumped whenever OBS_FIELDS changes; the manifest
# records it and the loader refuses mismatched shards rather than silently
# mis-shaping a batch.
SCHEMA_VERSION = 2

# The four ICRA benchmark objects, in a CANONICAL order that is independent of
# both the scene build and the log trace's spawn order. The one-hot is keyed on
# the object NAME through this map -- never on an index, because
# EpisodeMeta.obj_idx lives in the trace's spawn frame and disagrees with the
# scene's body order on 3 of these 4 (see extract._trace_object_index).
ICRA_OBJECT_NAMES = (
    "obj_036_wood_block",
    "obj_009_gelatin_box",
    "obj_017_orange",
    "obj_025_mug",
)
OBJ_INDEX = {name: i for i, name in enumerate(ICRA_OBJECT_NAMES)}

# Body-frame mesh AABB half-extents, metres.
#
# Deliberately a constant table rather than read from the model or the logs:
#
#   * `model.geom_size` is per-GEOM, and the mug has 43 collision hulls -- its
#     primary geom is a 1.2 cm sliver of the handle, which would tell a policy
#     the mug is the SMALLEST of the four objects when it is the widest. Even for
#     the single-hull objects the mesh frame is rotated relative to the body
#     frame, so the gelatin box's axes come out permuted.
#   * `events.jsonl`'s `size` field is that same unusable value, plus up to
#     4.5e-5 m of drift between MuJoCo versions (source pins 3.6, we run 3.14),
#     so it cannot be bit-matched from the live model anyway.
#
# Computed once by transforming each hull's vertices through geom_pos/geom_quat
# and taking the body-frame extent.
OBJ_SIZE = {
    "obj_036_wood_block":  (0.05196, 0.05173, 0.10300),
    "obj_009_gelatin_box": (0.04470, 0.05055, 0.01504),
    "obj_017_orange":      (0.03608, 0.03699, 0.03568),
    "obj_025_mug":         (0.05897, 0.04700, 0.04121),
}

# Nominal YCB masses, kg (model.body_mass). Constant per object in the
# nominal-only dataset; the ablation sweep scales these, which the rollout env
# cannot reproduce, so ablation episodes are excluded from extraction.
OBJ_MASS = {
    "obj_036_wood_block": 0.729,
    "obj_009_gelatin_box": 0.097,
    "obj_017_orange": 0.047,
    "obj_025_mug": 0.118,
}


# Optional phase one-hot, appended after the stateless dims so a dataset built
# with it stays a strict superset of one built without. OFF by default -- see
# OPTIONAL_OBS_FIELDS and the caveats in scripts/test_phase_onehot.py.
PHASES = ("APPROACH", "PICK", "TRANSPORT", "PLACE")
PHASE_INDEX = {p: i for i, p in enumerate(PHASES)}


def phase_onehot(name: str) -> list[float]:
    """One-hot over the trial phases. All-zero for an unknown label."""
    v = [0.0] * len(PHASES)
    i = PHASE_INDEX.get(str(name))
    if i is not None:
        v[i] = 1.0
    return v


def obj_onehot(name: str) -> list[float]:
    """One-hot over the ICRA objects, by name. All-zero for anything else."""
    v = [0.0] * len(ICRA_OBJECT_NAMES)
    i = OBJ_INDEX.get(name)
    if i is not None:
        v[i] = 1.0
    return v

# Observation layout. Order matters: it is the on-disk contract for every
# cached shard and every trained checkpoint.
OBS_FIELDS: tuple[tuple[str, int], ...] = (
    ("q_arm", N_ARM),            # rad
    ("q_hand", N_HAND),          # rad
    ("qd_arm", N_ARM),           # rad/s
    ("qd_hand", N_HAND),         # rad/s
    ("obj_in_palm", 3),          # m, palm frame
    ("obj_quat_in_palm", 4),     # wxyz, palm->object
    ("palm_pos", 3),             # m, world
    ("palm_quat", 4),            # wxyz, world
    ("norm_force", 4),           # N, per finger (thumb,index,middle,ring)
    ("tan_force", 4),            # N
    # --- schema v2 additions (appended, so every v1 slice stays valid) ---
    # Object conditioning. Without it the finger action is UNIDENTIFIABLE: only
    # 32% of matched-state neighbour pairs shared the query's object, and hand
    # action disagreement plateaued at 0.035-0.058 rad even at essentially
    # identical observations (vs the arm, whose disagreement -> 0).
    ("obj_id_onehot", 4),        # by name, via OBJ_INDEX
    ("obj_size", 3),             # m, body-frame AABB half-extents
    ("obj_mass", 1),             # kg
    # Task-stage proxy. Deliberately NOT the 4-way phase one-hot: PLACE is one
    # row per episode whose object features are already torn down, PICK is absent
    # from 115/222 episodes depending on the TELEOP MODE (so it leaks the
    # collection method), and APPROACH->PICK has no physical correlate -- it is a
    # human keypress. These two are stateless and reproducible on both sides.
    ("height_above_rest", 1),    # m, object lift above its resting height
    ("hand_touching", 1),        # 0/1, any fingertip in contact with the object
)
# Optional, opt-in blocks appended AFTER the fixed ones. A dataset built with
# them is a strict superset, so obs_slice() of every fixed field is unchanged.
OPTIONAL_OBS_FIELDS: dict[str, int] = {"phase_onehot": len(PHASES)}


def obs_dim(extras: tuple[str, ...] = ()) -> int:
    return (sum(n for _, n in OBS_FIELDS)
            + sum(OPTIONAL_OBS_FIELDS[e] for e in extras))


OBS_DIM = obs_dim()  # 78 at SCHEMA_VERSION 2 (was 68 at v1)

# Action spaces. Both are 23-d; they differ in units and in what the rollout
# env does with them. See handil/data/actions.py for how each is recovered.
#
#   "position" -- target joint position (rad), the semantic command the teleop
#       P-controller consumes. Recovered exactly from the logged torque. The
#       default, and the right choice for the baseline (raw teleop) pipelines.
#
#   "torque" -- the commanded joint torque (N*m) itself, net of gravity
#       compensation: qfrc_applied[:23] - qfrc_bias[:23]. Logged directly, so it
#       needs no inversion and no knowledge of the controller's gains.
#
# Why "torque" matters for the contact-aware pipeline: during its GRASP phase the
# commanded torque is
#
#     tau = kp_eff*(q_hold - qpos) + kd_eff*(qdot_cmd - qvel) + tau_internal + tau_slip
#
# where kp_eff is a time-varying product of squeeze_pd_scale, transport_pd_scale
# and gamma/gamma_ref (grasp_control/grasp_controller.py:effective_gains) that is
# NOT logged. So the position target is not recoverable there, while the torque
# is logged outright. A torque policy can express a squeeze; a position policy
# reconstructed by our GRASP repair cannot.
#   "delta" -- the same command expressed as (target - qpos), rad.
#
# Why "delta" matters, measured: corr(action[t], qpos[t]) = 0.91 and
# |action - qpos| is only 0.016 rad median, so with ABSOLUTE targets the
# identity map ("echo your current joint position") is a near-optimal regression
# solution. A BC-RNN trained that way predicts the demonstrated target to
# 0.0018 rad teacher-forced yet commands only 13-39% of the joint travel the
# task needs, so the robot barely moves and stalls ~0.33 m short of the object.
# Predicting the delta removes the shortcut: the residual IS the signal, and a
# policy that outputs zero now moves nothing instead of scoring well.
ACTION_SPACES = ("position", "torque", "delta")
ACTION_UNITS = {"position": "rad", "torque": "N*m", "delta": "rad"}
ACT_DIM = N_ROBOT

_OBS_SLICES: dict[str, slice] = {}
_off = 0
for _name, _n in OBS_FIELDS:
    _OBS_SLICES[_name] = slice(_off, _off + _n)
    _off += _n


def obs_slice(name: str) -> slice:
    """Column slice of `name` within a packed observation vector."""
    return _OBS_SLICES[name]


def obs_field_names() -> list[str]:
    """Per-column names, for feature-importance reporting and debugging."""
    out: list[str] = []
    for name, n in OBS_FIELDS:
        if n == 1:
            out.append(name)
        else:
            out.extend(f"{name}[{i}]" for i in range(n))
    return out


def split_obs(obs: np.ndarray) -> dict[str, np.ndarray]:
    """Unpack a packed (..., OBS_DIM) array back into named components."""
    if obs.shape[-1] != OBS_DIM:
        raise ValueError(f"expected last dim {OBS_DIM}, got {obs.shape[-1]}")
    return {name: obs[..., sl] for name, sl in _OBS_SLICES.items()}
