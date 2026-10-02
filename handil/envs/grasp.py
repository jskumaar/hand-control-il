"""Internal-force grasp torques for the rollout env.

This is what lets a *position* policy express a contact-aware squeeze. The
source's contact-aware GRASP phase commands

    tau = kp_eff*(q_hold - qpos) + kd_eff*(qdot_cmd - qvel)
          + tau_internal + tau_slip

where `tau_internal` is a null-space (pure internal) contact force allocated
over the live grasp map and mapped to joint torques through the fingertip
Jacobians. A plain P controller has no way to produce it: a position target
can close the fingers until they touch, but it cannot press.

Rather than reimplement that allocation and risk drifting from the source, this
module **imports the source repo's own `GraspController`** and drives it. Only
the squeeze is taken (`internal_force_torques`, and optionally
`slip_correction_torques`); the position hold stays the env's own P law, which
is exactly how `kinova_leap_pick_place.py` uses it in the teleop GRASP branch --
the method's docstring explicitly supports callers that own their PD.

Two notes on fidelity:

  * `grasp_control/__init__.py` imports `constrained_ik`, which needs casadi.
    The submodules are loaded directly to avoid that dependency.
  * The source's teleop path feeds the controller the NLP recommender's stored
    contact frames (`rec_local`), which are not in the logs. Here the contacts
    are read from the **live** fingertip/object geometry instead, so the squeeze
    is defined against where the fingers actually are. That makes this a faithful
    internal-force controller but not a replay of a specific recommended grasp.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

# Fingertip site names, from kinova_common/constants.py.
FINGER_TIP_SITES = {
    "index": "leap_if_ds_tip",
    "middle": "leap_mf_ds_tip",
    "ring": "leap_rf_ds_tip",
    "thumb": "leap_th_ds_tip",
}
FINGER_CODE = {"index": "if", "middle": "mf", "ring": "rf", "thumb": "th"}

# Finger-joint qpos slices on the LEAP hand (23-DOF layout: 7 arm + 16 hand).
FINGER_JOINT_SLICES = {
    "index": (7, 11), "middle": (11, 15), "ring": (15, 19), "thumb": (19, 23),
}

# Source defaults worth matching (kinova_leap_pick_place.py).
SQUEEZE_PD_SCALE = 0.25
SQUEEZE_RAMP_S = 0.5
CONE_MARGIN = 0.2
CONE_F_MIN = 0.5


def load_grasp_controller(root: str | Path):
    """Import `GraspController` from the hand-control repo.

    Loads `grasp_control.{grasp_map,force_control,grasp_controller}` directly,
    bypassing the package `__init__` (which pulls in casadi via constrained_ik).
    """
    root = Path(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if "grasp_control.grasp_controller" not in sys.modules:
        pkg = types.ModuleType("grasp_control")
        pkg.__path__ = [str(root / "grasp_control")]
        sys.modules.setdefault("grasp_control", pkg)
        for name in ("grasp_map", "force_control", "grasp_controller"):
            full = f"grasp_control.{name}"
            if full in sys.modules:
                continue
            spec = importlib.util.spec_from_file_location(
                full, root / "grasp_control" / f"{name}.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[full] = mod
            spec.loader.exec_module(mod)
    from grasp_control.grasp_controller import GraspController  # noqa: E402
    return GraspController


def tip_site_ids(model, fingers: tuple[str, ...]):
    import mujoco as mj
    out = []
    for f in fingers:
        sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, FINGER_TIP_SITES[f])
        if sid < 0:
            raise KeyError(f"fingertip site for {f} not in model")
        out.append(int(sid))
    return out


def make_live_contact_provider(model, obj_body_id: int, tip_sids: list[int]):
    """Contact frames from the live fingertip/object geometry.

    Returns `provider(data) -> [(p_W, R_W), ...]`, one per finger, where `p_W` is
    the contact point and `R_W`'s column 0 is the INWARD surface normal -- the
    convention the controller's contact sites use.

    The normal is taken as the direction from the fingertip toward the object
    body origin, which is exact for a convex object grasped near its centre and
    is the same approximation the source's sign anchor makes on its first
    contact.
    """
    def provider(data):
        p_obj = data.xpos[obj_body_id]
        out = []
        for sid in tip_sids:
            p_tip = data.site_xpos[sid]
            d = p_obj - p_tip
            n = np.linalg.norm(d)
            inward = d / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
            # Any orthonormal frame whose col0 is the inward normal.
            helper = (np.array([0.0, 0.0, 1.0])
                      if abs(inward[2]) < 0.9 else np.array([1.0, 0.0, 0.0]))
            t1 = np.cross(inward, helper)
            t1 /= max(np.linalg.norm(t1), 1e-9)
            t2 = np.cross(inward, t1)
            out.append((p_tip.copy(), np.column_stack([inward, t1, t2])))
        return out
    return provider


class SqueezeController:
    """Wraps the source `GraspController` to supply only the squeeze torque.

    Usage per control step:

        sq.update(data)                      # refresh contacts / ramp
        tau[:23] += sq.torques(data)         # add to the env's own P torque

    `gamma` is the per-contact internal normal force in newtons. The source
    solves it per grasp; a fixed value is used here, and `gamma_from_force` maps
    a target grip force onto it.
    """

    def __init__(self, model, obj_body_id: int, kp: np.ndarray, kd: np.ndarray,
                 fingers: tuple[str, ...] = ("thumb", "index"),
                 gamma: float = 5.0, root: str | Path = None,
                 obj_geom_ids=None, cone_mu: float = None,
                 slip_correction: bool = True,
                 squeeze_pd_scale: float = SQUEEZE_PD_SCALE,
                 ramp_s: float = SQUEEZE_RAMP_S, dt: float = 0.002):
        from .scene import handcontrol_root

        GC = load_grasp_controller(handcontrol_root(root))
        self.fingers = tuple(fingers)
        self.tip_sids = tip_site_ids(model, self.fingers)
        self.slices = tuple(FINGER_JOINT_SLICES[f] for f in self.fingers)
        self.slip_correction = slip_correction
        self.ramp_s, self.dt = ramp_s, dt
        self._steps = 0

        if cone_mu is None:
            cone_mu = 0.7
        self.ctrl = GC(
            model, 23,
            tip_site_ids=self.tip_sids,
            obj_site_ids=None,
            obj_body_id=int(obj_body_id),
            kp=np.asarray(kp, float), kd=np.asarray(kd, float),
            gamma=float(gamma),
            squeeze_pd_scale=squeeze_pd_scale,
            support_weight=True,
            obj_contact_provider=make_live_contact_provider(
                model, int(obj_body_id), self.tip_sids),
            cone_gamma=True, cone_mu=float(cone_mu),
            cone_margin=CONE_MARGIN, cone_f_min=CONE_F_MIN,
            active_joint_slices=self.slices,
            obj_geom_ids=obj_geom_ids,
        )
        self.ctrl.set_squeeze(True)

    @property
    def ramp(self) -> float:
        """0 -> 1 over `ramp_s`; full force on a half-closed grasp shoves it out."""
        return min(1.0, self._steps * self.dt / self.ramp_s)

    def reset(self):
        self._steps = 0
        self.ctrl.set_squeeze(True)
        self.ctrl.set_transporting(False)
        self.ctrl.last_f_c = None

    def set_transporting(self, on: bool):
        """Switch CLOSING -> HOLDING finger gains once the squeeze has arrived."""
        self.ctrl.set_transporting(bool(on))

    def torques(self, data) -> np.ndarray:
        """Internal-force (+ slip) joint torque for the 23 robot DOFs."""
        self._steps += 1
        tau = np.zeros(23)
        try:
            tau += self.ctrl.internal_force_torques(data, scale=self.ramp)
            if self.slip_correction:
                tau += self.ctrl.slip_correction_torques(data)
        except Exception:
            # A degenerate grasp map (no live contacts, singular null space) is
            # normal before the fingers touch; contribute nothing those steps.
            return np.zeros(23)
        return np.nan_to_num(tau, nan=0.0, posinf=0.0, neginf=0.0)
