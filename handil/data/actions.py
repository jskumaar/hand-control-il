"""Recovering action labels from hand-control logs.

The source repo never logs `_dp_target`, the 23-vector of target joint
positions that is the semantic action. It logs the *consequence*: the torque
actually written to `data.qfrc_applied`. The controller is

    tau_ctrl[:7]  = Kp_arm * (target[:7] - qpos[:7]) + Kd_arm * (qdot_cmd - qvel[:7])
    tau_ctrl[7:]  = Kp_hand * (target[7:] - qpos[7:])
    qfrc_applied[:23] = tau_ctrl[:23] + qfrc_bias[:23]        # gravity comp

(kinova_leap_pick_place.py:6452-6508). Two facts make this exactly invertible:

  * The arm Kd term is identically zero by construction. The drive sets
    `data.qvel[:7] = _qdot_arm` on the line before it is evaluated, so
    `(qdot_cmd - qvel[:7])` is 0. Real damping lives in `model.dof_damping`.
  * Gravity compensation uses `qfrc_bias`, which is logged.

  =>  target = qpos + (qfrc_applied - qfrc_bias) / Kp

The gains are known analytically rather than fitted:

  * Arm: `diag(M)[:7] * WN**2`, with M the full mass matrix evaluated at the
    wrist-down home pose (kinova_leap_pick_place.py:3626-3635). Constant for a
    given scene, so we compute it once from the model.
  * Hand: a single scalar selected by collection mode -- 0.8 for the
    contact-aware pipeline (`Kp[7:]`, :2128) and 2.0 for the plain dexpilot /
    anyteleop / vwj baselines (`KP_TELEOP_FINGER`, :2138).

An earlier version of this module tried to *fit* the gains from trajectory
smoothness. It was validated against the analytic arm gain and was wrong by up
to three orders of magnitude, so it was removed. Gains come from the model.
"""
from __future__ import annotations

import numpy as np

from .schema import N_ARM, N_ROBOT

WN = 30.0  # rad/s, the arm's target natural frequency

# Hand proportional gain by collection mode. The contact-aware pipeline runs a
# soft finger gain and adds internal-force torques on top; the baselines are
# position-only with a stiffer gain.
KP_HAND_BY_MODE: dict[str, float] = {
    "contact_aware_teleop": 0.8,
    "contact_aware_w_dexpilot": 0.8,
    "contact_aware_w_anyteleop": 0.8,
    "contact_aware_w_vwj": 0.8,
    "contact_aware_w_vwj_upstream": 0.8,
    "contact_aware_autonomous": 0.8,
    "rrt": 0.8,
    "dexpilot": 2.0,
    "anyteleop": 2.0,
    "vwj": 2.0,
    "vwj_upstream": 2.0,
}
KP_HAND_DEFAULT = 2.0

# Fallback arm gains for when the MuJoCo model is unavailable. Computed from
# scene_pick_place.xml at the wrist-down home; see tests/test_actions.py.
KP_ARM_FALLBACK = np.array([509.3, 675.5, 254.3, 324.0, 65.9, 65.0, 3.1])


def kp_hand_for_mode(mode: str | None) -> float:
    """Finger gain for a collection mode, defaulting to the baseline value."""
    if mode is None:
        return KP_HAND_DEFAULT
    return KP_HAND_BY_MODE.get(mode, KP_HAND_DEFAULT)


def kp_arm_from_model(model, home_qpos: np.ndarray | None = None) -> np.ndarray:
    """Arm gains as the source repo computes them: diag(M)[:7] * WN**2.

    Reproduces kinova_leap_pick_place.py:3626-3635 exactly, including the
    1e-3 inertia floor and the evaluation at the wrist-down home pose.
    """
    import mujoco as mj

    from ..envs.scene import full_mass_matrix

    data = mj.MjData(model)
    if home_qpos is not None:
        data.qpos[: len(home_qpos)] = home_qpos
    mj.mj_forward(model, data)
    inertia = np.clip(np.diag(full_mass_matrix(model, data))[:N_ARM], 1e-3, None)
    return inertia * WN**2


def targets_from_torque(qpos: np.ndarray, qfrc_applied: np.ndarray,
                        qfrc_bias: np.ndarray, kp_arm: np.ndarray,
                        kp_hand: float) -> np.ndarray:
    """Invert the P law to commanded joint targets. Returns (N, 23) rad."""
    kp = np.concatenate([
        np.asarray(kp_arm, float).reshape(N_ARM),
        np.full(N_ROBOT - N_ARM, float(kp_hand)),
    ])
    tau = qfrc_applied[:, :N_ROBOT] - qfrc_bias[:, :N_ROBOT]
    return qpos[:, :N_ROBOT] + tau / kp


def torques_from_log(qfrc_applied: np.ndarray,
                     qfrc_bias: np.ndarray) -> np.ndarray:
    """The commanded joint torque, net of gravity compensation. (N, 23) N*m.

    No inversion and no gains needed -- this is what the controller actually
    wrote, minus the `qfrc_bias` it added for gravity compensation:

        qfrc_applied[:23] = tau_ctrl[:23] + qfrc_bias[:23]

    Use this action space for the contact-aware pipeline, whose GRASP-phase
    position target is NOT recoverable (its kp_eff is a time-varying product of
    squeeze/transport scales and gamma/gamma_ref that the logs do not carry),
    while its torque is logged outright and includes the internal-force squeeze.
    """
    return qfrc_applied[:, :N_ROBOT] - qfrc_bias[:, :N_ROBOT]


def targets_from_logged_grasp(qpos: np.ndarray, qfrc_applied: np.ndarray,
                             qfrc_bias: np.ndarray, kp_arm: np.ndarray,
                             kp_hand: float,
                             q_grasp_hold: np.ndarray | None,
                             kp_eff: np.ndarray | None,
                             control_phase: np.ndarray | None
                             ) -> tuple[np.ndarray, float]:
    """Exact targets when the run logged the GRASP-phase hold and gains.

    Preferred over `blend_grasp_phase` whenever the trace carries `q_grasp_hold`
    and `kp_eff` (added to hand-control for this reason -- see
    docs/BEFORE_COLLECTING.md). The GRASP torque is

        kp_eff*(q_grasp_hold - qpos) + kd_eff*(...) + tau_internal + tau_slip

    so with the hold logged directly there is nothing to invert on those rows:
    the commanded target IS `q_grasp_hold`. No repair, no substituted
    `qpos[t+1]`, and the squeeze the fingers were commanding survives into the
    label.

    Returns (targets, fraction_of_rows_taken_from_the_log).
    """
    a = targets_from_torque(qpos, qfrc_applied, qfrc_bias, kp_arm, kp_hand)
    if q_grasp_hold is None or control_phase is None:
        return a, 0.0
    hold = np.asarray(q_grasp_hold, float)
    if hold.ndim != 2 or hold.shape[0] != a.shape[0]:
        return a, 0.0
    # Rows where the hold is real (it is NaN outside GRASP).
    live = np.isfinite(hold[:, :N_ROBOT]).all(axis=1)
    if not live.any():
        return a, 0.0
    out = a.copy()
    out[live] = hold[live, :N_ROBOT]
    return out, float(live.mean())


def blend_grasp_phase(targets: np.ndarray, qpos: np.ndarray,
                      control_phase: np.ndarray) -> tuple[np.ndarray, float]:
    """Repair finger targets during the contact-aware GRASP phase.

    In `control_phase == 'GRASP'` the contact-aware pipeline adds internal-force
    and slip-correction torques on top of the joint PD term
    (kinova_leap_pick_place.py:8541, :8563). `qfrc_applied` is then no longer
    `Kp * err`, so inverting it over-states the finger target badly -- measured
    at up to 20 rad, far outside the LEAP joint limits, versus 0.017 rad during
    REACH where the drive is pure PD.

    The arm is unaffected (those torques act on the hand), so only the 16 finger
    columns are replaced, with the next achieved finger position. During a
    closing grasp the fingers are contact-blocked, which makes `qpos[t+1]`
    conservative -- but a conservative label beats an unphysical one, and the
    squeeze that the extra torque encodes is not representable as a position
    target anyway.

    Returns (repaired_targets, fraction_of_rows_repaired).
    """
    if control_phase is None or len(control_phase) != len(targets):
        return targets, 0.0
    phase = np.asarray(control_phase).astype(str)
    mask = phase == "GRASP"
    if not mask.any():
        return targets, 0.0
    out = targets.copy()
    nxt = targets_from_next_qpos(qpos)
    out[mask, N_ARM:] = nxt[mask, N_ARM:]
    return out, float(mask.mean())


def targets_from_next_qpos(qpos: np.ndarray) -> np.ndarray:
    """Fallback label: the next achieved joint position.

    Standard for position-controlled BC, and the only option for runs whose
    qfrc_applied / qfrc_bias were not logged (the narrower legacy schemas).
    Under-states the command whenever a joint is blocked by contact -- which is
    exactly what happens during the grasp -- so prefer `targets_from_torque`.
    """
    nxt = np.empty_like(qpos[:, :N_ROBOT])
    nxt[:-1] = qpos[1:, :N_ROBOT]
    nxt[-1] = qpos[-1, :N_ROBOT]
    return nxt


def deltas_from_targets(targets: np.ndarray, qpos: np.ndarray) -> np.ndarray:
    """Express joint targets as a residual on the current position. (N, 23) rad.

    The inverse, applied at rollout, is `qpos + delta`. See the ACTION_SPACES
    note in schema.py for why this parameterisation is not cosmetic here.
    """
    return targets[:, :N_ROBOT] - qpos[:, :N_ROBOT]


def bounded_mask(jnt_range: np.ndarray,
                 jnt_limited: np.ndarray | None = None) -> np.ndarray:
    """Which of the 23 joints have a real physical stop.

    `jnt_limited` is authoritative when given -- MuJoCo's own flag. Without it,
    fall back to `hi > lo`, the convention an unlimited joint is stored under.
    Arm joints 0/2/4/6 are continuous and must come out False, or clipping
    against their nominal range silently destroys the trajectory.
    """
    if jnt_limited is not None:
        return np.asarray(jnt_limited[:N_ROBOT]).astype(bool)
    return jnt_range[:N_ROBOT, 1] > jnt_range[:N_ROBOT, 0]


def clip_to_limits(targets: np.ndarray, jnt_range: np.ndarray,
                   jnt_limited: np.ndarray | None = None) -> np.ndarray:
    """Clip recovered targets into the joint limits.

    Two sources of out-of-range targets, neither meaningful to imitate:

      * Low-gain joints. The wrist roll has Kp=3.1 against 509 at the shoulder,
        so a small logged torque divides into a large implied error. Joint 6
        accounts for essentially all of the residual violation.
      * Continuous joints. The source clamps joints 0/2/4/6 to +-pi
        (kinova_leap_pick_place.py:767) even though they rotate freely, so a
        target past pi is a wrap artifact rather than a real command.

    The rollout env clips identically before applying a target, so clipping
    here keeps the training labels and the executed actions in the same space.
    """
    lo, hi = jnt_range[:N_ROBOT, 0], jnt_range[:N_ROBOT, 1]
    bounded = bounded_mask(jnt_range, jnt_limited)
    out = targets.copy()
    out[:, bounded] = np.clip(out[:, bounded], lo[bounded], hi[bounded])
    return out


def sanity_report(targets: np.ndarray, jnt_range: np.ndarray | None = None,
                  jnt_limited: np.ndarray | None = None) -> dict:
    """Cheap diagnostics on recovered targets, for the extraction log.

    A target stream that leaves the joint limits by a wide margin is the
    signature of a wrong gain, so this is worth printing per run.
    """
    out = {
        "n": int(targets.shape[0]),
        "min": float(targets.min()),
        "max": float(targets.max()),
        "step_p95_rad": float(np.percentile(np.abs(np.diff(targets, axis=0)), 95)),
    }
    if jnt_range is not None:
        lo = jnt_range[:N_ROBOT, 0]
        hi = jnt_range[:N_ROBOT, 1]
        bounded = bounded_mask(jnt_range, jnt_limited)
        viol = np.zeros_like(targets)
        viol[:, bounded] = np.maximum(
            lo[bounded] - targets[:, bounded], targets[:, bounded] - hi[bounded]
        ).clip(min=0.0)
        out["limit_viol_p99_rad"] = float(np.percentile(viol, 99))
        out["limit_viol_max_rad"] = float(viol.max())
    return out
