"""Closed-loop policy rollout in the hand-control MuJoCo scene.

The controller is copied from the source repo's drive loop
(kinova_leap_pick_place.py:6452-6508) with the parts that only make sense for
a human operator removed. Specifically:

  * Arm velocity injection (`data.qvel[:7] = _qdot_arm`) is NOT reproduced.
    It exists so a teleoperator's Cartesian jog rate rides open-loop without
    inertial lag; a policy emits joint *targets*, not rates, so injecting a
    velocity it never commanded would fight its own position target. The
    logged demos were produced with injection on, which is a genuine
    train/rollout mismatch -- it is called out in the README rather than
    papered over, and `velocity_injection=True` is available for A/B testing.
  * The arm Kd term is dropped because it is identically zero in the source
    (see handil/data/actions.py). Damping comes from `model.dof_damping`,
    which build_scene() sets to the same 2*I*wn.

Everything else is bit-for-bit: no torque clamping (deliberately -- the source
has none), the same qvel clamp, the same `qfrc_applied = tau + qfrc_bias` with
gravity compensation on the robot DOFs only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data.schema import N_ARM, N_ROBOT, OBS_DIM
from .scene import (SEQ_SPAWN_XY, STOW_Z, Scene, build_scene, home_qpos)

QVEL_ARM_MAX = 4.0        # rad/s, matches the source safety clamp

# Success criteria (trial_logger.py:47-101).
LIFT_HEIGHT_M = 0.01
DWELL_S = 1.0
ARRIVAL_SPEED_M_S = 0.05
ARRIVAL_DWELL_S = 0.5
DROP_DWELL_S = 0.3
IN_BOWL_GRACE_S = 0.2


@dataclass
class RolloutResult:
    success: bool
    outcome: str                  # success | timeout | dropped
    steps: int
    sim_time: float
    max_height: float
    picked: bool
    n_drops: int
    obs: np.ndarray = field(default=None, repr=False)
    action: np.ndarray = field(default=None, repr=False)


class PickPlaceEnv:
    """Minimal env exposing the same obs/action contract as the dataset."""

    def __init__(self, scene: Scene | None = None, kp_hand: float = 2.0,
                 control_hz: float = 42.0, velocity_injection: bool = False,
                 scene_name: str = "robocasa", root=None,
                 active_object: int | str = 0, sequential: bool = True,
                 action_space: str = "position", squeeze: bool = False,
                 gamma: float = 5.0, fingers=("thumb", "index"),
                 slip_correction: bool = True, root_for_grasp=None):
        import mujoco as mj
        self.mj = mj
        self.scene = scene if scene is not None else build_scene(scene_name, root=root)
        self.model = self.scene.model
        self.data = self.scene.data
        self.kp = np.concatenate([self.scene.kp_arm,
                                  np.full(N_ROBOT - N_ARM, float(kp_hand))])
        self.velocity_injection = velocity_injection
        if action_space not in ("position", "torque"):
            raise ValueError(f"action_space must be position|torque, got {action_space!r}")
        self.action_space = action_space
        self._tau_cmd = np.zeros(N_ROBOT)
        # Internal-force squeeze. A P law can close the fingers but cannot press,
        # so a contact-aware grasp is inexpressible without this. See
        # handil/envs/grasp.py.
        self.squeeze_enabled = bool(squeeze)
        self._squeeze_cfg = dict(gamma=gamma, fingers=tuple(fingers),
                                 slip_correction=slip_correction,
                                 root=root_for_grasp)
        self.squeeze = None
        # Demos are logged at ~42 Hz wall; the policy runs at that rate and each
        # control step is held for the matching number of 2 ms physics steps.
        self.n_substeps = max(1, int(round((1.0 / control_hz) / self.model.opt.timestep)))
        # Collision masks as compiled, captured by build_scene before any env
        # stowed into this model. A Scene can be shared between env instances, so
        # restore them here instead of reading the (possibly zeroed) live values.
        self._contype0 = np.asarray(self.scene.contype0).copy()
        self._conaff0 = np.asarray(self.scene.conaff0).copy()
        self.model.geom_contype[:] = self._contype0
        self.model.geom_conaffinity[:] = self._conaff0

        from ..data.actions import bounded_mask
        self._bounded = bounded_mask(np.asarray(self.model.jnt_range),
                                     np.asarray(self.model.jnt_limited))

        self._bowl = self._bowl_geometry()
        self._obj_col_gids = self._object_geoms()
        self._in_bowl_t: dict[int, float] = {}
        self.sequential = sequential
        self._col0 = {i: [(g, int(self._contype0[g]),
                           int(self._conaff0[g]))
                          for g in gids]
                      for i, gids in self._obj_col_gids.items()}
        self.set_active_object(active_object)
        if self.squeeze_enabled:
            self._build_squeeze()
        self.reset()

    def _build_squeeze(self):
        """Attach an internal-force controller to the ACTIVE object."""
        from .grasp import SqueezeController
        cfg = self._squeeze_cfg
        bid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY,
                                 self.scene.object_bodies[self.active_object])
        gids = self._obj_col_gids[self.active_object]
        kd = np.concatenate([np.full(N_ARM, 4.0),
                             np.full(N_ROBOT - N_ARM, 0.05)])
        self.squeeze = SqueezeController(
            self.model, bid, self.kp, kd, fingers=cfg["fingers"],
            gamma=cfg["gamma"], root=cfg["root"], obj_geom_ids=gids,
            cone_mu=float(self.model.geom_friction[gids[0], 0]),
            slip_correction=cfg["slip_correction"],
            dt=float(self.model.opt.timestep))

    def set_active_object(self, which: int | str) -> int:
        """Choose which object is presented; the rest are stowed.

        Accepts an index or a body name ('obj_025_mug') or a YCB id fragment
        ('025_mug'), since episode metadata names objects, not indices.
        """
        names = self.scene.object_bodies
        if isinstance(which, str):
            match = [i for i, n in enumerate(names) if n == which]
            if not match:
                match = [i for i, n in enumerate(names) if which.lstrip("obj_") in n]
            if not match:
                raise KeyError(f"no object matching {which!r} in {names}")
            which = match[0]
        if not 0 <= which < len(names):
            raise IndexError(f"object index {which} out of range for {names}")
        self.active_object = int(which)
        return self.active_object

    # -- scene queries ----------------------------------------------------
    def _bowl_geometry(self):
        mj = self.mj
        base = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "bowl_base")
        if base < 0:
            return None
        gids = set()
        for n in ("bowl_base", "bowl_wall_n", "bowl_wall_s", "bowl_wall_e", "bowl_wall_w"):
            g = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, n)
            if g >= 0:
                gids.add(g)
        c = self.data.geom_xpos[base].copy()
        return {"xy": c[:2].copy(), "hx": float(self.model.geom_size[base][0]),
                "hy": float(self.model.geom_size[base][1]),
                "base_gid": base, "gids": gids}

    def _object_geoms(self) -> dict[int, list[int]]:
        """Collision geoms per object; YCB meshes are multi-hull (mug = 43).

        Built from a snapshot of the AS-COMPILED contype taken before any
        stowing, since stowing zeroes contype -- reading it live would return an
        empty geom list for every stowed object.
        """
        if not hasattr(self, "_contype0"):
            self._contype0 = self.model.geom_contype.copy()
        out = {}
        for i, name in enumerate(self.scene.object_bodies):
            bid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, name)
            gl = [g for g in range(self.model.ngeom)
                  if self.model.geom_bodyid[g] == bid and self._contype0[g] != 0]
            out[i] = gl
        return out

    def _center_over_tray(self, i: int, margin: float = 0.03) -> bool:
        if not self._bowl:
            return False
        gid = self._obj_col_gids[i][0]
        p = self.data.geom_xpos[gid]
        return (abs(float(p[0]) - self._bowl["xy"][0]) <= self._bowl["hx"] + margin
                and abs(float(p[1]) - self._bowl["xy"][1]) <= self._bowl["hy"] + margin)

    def object_in_bowl(self, i: int) -> bool:
        """Contact-based containment, as the source scores it.

        Deliberately contact-with-`bowl_base`, not a z-band: walls do not
        count, and there is a 0.2 s flicker latch. No penetration filter --
        the source found resting interpenetration (0.018 mm) is always below
        any sensible threshold, which broke placement entirely.
        """
        if not self._bowl or not self._center_over_tray(i):
            return self._latched(i)
        mine = set(self._obj_col_gids[i])
        support = {self._bowl["base_gid"]}
        added = True
        while added:                       # transitive support through stacks
            added = False
            for j in range(len(self.scene.object_bodies)):
                if j == i or not self._center_over_tray(j):
                    continue
                jg = set(self._obj_col_gids[j])
                if jg & support:
                    continue
                for ci in range(self.data.ncon):
                    c = self.data.contact[ci]
                    if ((c.geom1 in jg and c.geom2 in support)
                            or (c.geom2 in jg and c.geom1 in support)):
                        support |= jg
                        added = True
                        break
        for ci in range(self.data.ncon):
            c = self.data.contact[ci]
            if ((c.geom1 in mine and c.geom2 in support)
                    or (c.geom2 in mine and c.geom1 in support)):
                self._in_bowl_t[i] = float(self.data.time)
                return True
        return self._latched(i)

    def _latched(self, i: int) -> bool:
        last = self._in_bowl_t.get(i)
        return last is not None and (self.data.time - last) <= IN_BOWL_GRACE_S

    def hand_touching(self, i: int) -> bool:
        mine = set(self._obj_col_gids[i])
        for ci in range(self.data.ncon):
            c = self.data.contact[ci]
            for g, o in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if o in mine:
                    bid = self.model.geom_bodyid[g]
                    name = self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_BODY, bid) or ""
                    if name.startswith("leap_"):
                        return True
        return False

    # -- env API ----------------------------------------------------------
    def _park(self, i: int):
        """Hold object i's freejoint at its stow pose with zero velocity."""
        qa = self.scene.object_qadr[i]
        self.data.qpos[qa:qa + 3] = [0.3 * i, 0.0, STOW_Z]
        self.data.qpos[qa + 3:qa + 7] = [1.0, 0.0, 0.0, 0.0]
        bid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY,
                                 self.scene.object_bodies[i])
        dadr = int(self.model.jnt_dofadr[self.model.body_jntadr[bid]])
        self.data.qvel[dadr:dadr + 6] = 0.0

    def _stow(self, i: int):
        """Park object i below the floor with collision OFF.

        Collision must be disabled first: the source repo documents that a
        still-colliding body parked at z=-5 makes MuJoCo see a multi-metre floor
        interpenetration and fling the whole scene (observed |qvel| ~ 4400).
        """
        for g, _, _ in self._col0[i]:
            self.model.geom_contype[g] = 0
            self.model.geom_conaffinity[g] = 0
        self._park(i)

    def _activate(self, i: int):
        """Restore object i's collision and place it at the shared spawn point."""
        for g, ct, ca in self._col0[i]:
            self.model.geom_contype[g] = ct
            self.model.geom_conaffinity[g] = ca
        qa = self.scene.object_qadr[i]
        self.data.qpos[qa:qa + 2] = SEQ_SPAWN_XY
        self.data.qpos[qa + 3:qa + 7] = [1.0, 0.0, 0.0, 0.0]

    def reset(self) -> np.ndarray:
        mj = self.mj
        mj.mj_resetData(self.model, self.data)
        self.data.qpos[:N_ROBOT] = home_qpos()
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        self.data.qacc_warmstart[:] = 0.0

        if self.sequential:
            for i in range(len(self.scene.object_bodies)):
                if i != self.active_object:
                    self._stow(i)
            self._activate(self.active_object)
            mj.mj_forward(self.model, self.data)
            # Drop the active object onto the counter, as place_on_surface does.
            self._settle_active()

        mj.mj_forward(self.model, self.data)
        self._in_bowl_t.clear()
        self._rest_z = [float(self.data.xpos[
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, n)][2])
            for n in self.scene.object_bodies]
        self._target = self.data.qpos[:N_ROBOT].copy()
        self._tau_cmd = np.zeros(N_ROBOT)
        if self.squeeze is not None:
            self.squeeze.reset()
        return self.observe()

    def _settle_active(self, steps: int = 250):
        """Rest the active object on the counter surface.

        The source's `place_on_surface` works off the compiled spawn pose; here
        the object has just been teleported, so it is instead dropped from a few
        centimetres and allowed to settle under gravity with the robot frozen.
        Cheaper than computing a mesh AABB per object, and it handles the mug's
        43 collision hulls without special-casing.
        """
        import mujoco as mj

        i = self.active_object
        qa = self.scene.object_qadr[i]
        self.data.qpos[qa + 2] = self.scene.table_top_z + 0.05
        mj.mj_forward(self.model, self.data)

        hold = self.data.qpos[:N_ROBOT].copy()
        stowed = [j for j in range(len(self.scene.object_bodies)) if j != i]
        for _ in range(steps):
            self.data.qfrc_applied[:] = 0.0
            self.data.qpos[:N_ROBOT] = hold      # pin the robot while it settles
            self.data.qvel[:N_ROBOT] = 0.0
            # Stowed objects have collision off, so gravity would walk them
            # steadily downwards over every reset; re-park them each step.
            for j in stowed:
                self._park(j)
            mj.mj_step(self.model, self.data)
        self.data.qvel[:] = 0.0
        self.data.qacc_warmstart[:] = 0.0
        self.data.time = 0.0
        mj.mj_forward(self.model, self.data)

    def observe(self, obj_idx: int | None = None) -> np.ndarray:
        """Pack the same 68-d observation the dataset uses.

        Defaults to the ACTIVE object. Defaulting to index 0 instead would
        silently measure against a stowed object parked below the floor,
        producing |obj_in_palm| ~ 6 m and a policy input the demos never show.
        """
        if obj_idx is None:
            obj_idx = self.active_object
        mj = self.mj
        d, m = self.data, self.model
        q = d.qpos[:N_ROBOT].copy()
        qd = d.qvel[:N_ROBOT].copy()
        palm_p = d.xpos[self.scene.palm_bid].copy()
        palm_q = d.xquat[self.scene.palm_bid].copy()

        obj_p = np.full(3, np.nan)
        obj_q = np.full(4, np.nan)
        if self.scene.object_bodies:
            bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY,
                                self.scene.object_bodies[obj_idx])
            R = d.xmat[self.scene.palm_bid].reshape(3, 3)
            obj_p = R.T @ (d.xpos[bid] - palm_p)
            Rrel = R.T @ d.xmat[bid].reshape(3, 3)
            qr = np.zeros(4)
            mj.mju_mat2Quat(qr, Rrel.flatten())
            obj_q = qr

        nf, tf = self._finger_forces(obj_idx)
        return np.concatenate([
            q[:N_ARM], q[N_ARM:], qd[:N_ARM], qd[N_ARM:],
            obj_p, obj_q, palm_p, palm_q, nf, tf]).astype(np.float64)

    def _finger_forces(self, obj_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Per-finger normal/tangential contact force, order thumb,index,mf,rf."""
        mj = self.mj
        nf, tf = np.zeros(4), np.zeros(4)
        order = {"th": 0, "if": 1, "mf": 2, "rf": 3}
        mine = set(self._obj_col_gids.get(obj_idx, []))
        f6 = np.zeros(6)
        for ci in range(self.data.ncon):
            c = self.data.contact[ci]
            if not ({c.geom1, c.geom2} & mine):
                continue
            other = c.geom2 if c.geom1 in mine else c.geom1
            bid = self.model.geom_bodyid[other]
            name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, bid) or ""
            slot = next((v for k, v in order.items() if f"leap_{k}" in name), None)
            if slot is None:
                continue
            mj.mj_contactForce(self.model, self.data, ci, f6)
            nf[slot] += abs(float(f6[0]))
            tf[slot] += float(np.linalg.norm(f6[1:3]))
        return nf, tf

    def step(self, action: np.ndarray) -> np.ndarray:
        """Apply one control period of `action`.

        In "position" mode the action is a 23-vector of joint targets (rad) and a
        P law converts it to torque. In "torque" mode the action IS the commanded
        joint torque (N*m) and is applied directly -- the only faithful way to
        execute a contact-aware grasp, whose squeeze is an internal-force torque
        with no position-target equivalent.
        """
        a = np.asarray(action, float).reshape(N_ROBOT).copy()
        if self.action_space == "torque":
            self._tau_cmd = a
            self._target = None
        else:
            self._target = a
            # Clip only joints with a real stop: joints 0/2/4/6 are continuous
            # and the demonstrations drive them well past pi.
            b = self._bounded
            self._target[b] = np.clip(self._target[b],
                                      self.model.jnt_range[:N_ROBOT, 0][b],
                                      self.model.jnt_range[:N_ROBOT, 1][b])
        tau = np.zeros(self.model.nv)
        stowed = ([j for j in range(len(self.scene.object_bodies))
                   if j != self.active_object] if self.sequential else [])
        for _ in range(self.n_substeps):
            for j in stowed:          # collision is off, so gravity would drift them
                self._park(j)
            if self.velocity_injection:
                self.data.qvel[:N_ARM] = np.clip(
                    self.data.qvel[:N_ARM], -QVEL_ARM_MAX, QVEL_ARM_MAX)
            tau[:] = 0.0
            if self.action_space == "torque":
                tau[:N_ROBOT] = self._tau_cmd
            else:
                tau[:N_ROBOT] = self.kp * (self._target
                                           - self.data.qpos[:N_ROBOT])
            if self.squeeze is not None:
                tau[:N_ROBOT] += self.squeeze.torques(self.data)
            np.clip(self.data.qvel[:N_ARM], -QVEL_ARM_MAX, QVEL_ARM_MAX,
                    out=self.data.qvel[:N_ARM])
            self.data.qfrc_applied[:] = tau
            self.data.qfrc_applied[:N_ROBOT] += self.data.qfrc_bias[:N_ROBOT]
            self.mj.mj_step(self.model, self.data)
        return self.observe()

    def object_height_above_rest(self, i: int | None = None) -> float:
        if i is None:
            i = self.active_object
        bid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY,
                                 self.scene.object_bodies[i])
        return float(self.data.xpos[bid][2]) - self._rest_z[i]

    def object_speed(self, i: int | None = None) -> float:
        if i is None:
            i = self.active_object
        qadr = self.scene.object_qadr[i]
        bid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY,
                                 self.scene.object_bodies[i])
        jadr = self.model.body_jntadr[bid]
        dadr = int(self.model.jnt_dofadr[jadr])
        return float(np.linalg.norm(self.data.qvel[dadr:dadr + 3]))


def rollout(env: PickPlaceEnv, policy, max_steps: int = 600,
            obj_idx: int | None = None, record: bool = False) -> RolloutResult:
    """Run one episode and score it with the source repo's criteria."""
    obs = env.reset()
    if obj_idx is None:
        obj_idx = env.active_object
    obs_log, act_log = [], []
    dwell_t0 = drop_t0 = arrive_t0 = None
    picked = False
    n_drops = 0
    max_h = -np.inf

    for step in range(max_steps):
        a = np.asarray(policy(obs), float).reshape(N_ROBOT)
        if record:
            obs_log.append(obs.copy())
            act_log.append(a.copy())
        obs = env.step(a)
        t = float(env.data.time)

        h = env.object_height_above_rest(obj_idx)
        max_h = max(max_h, h)
        held = env.hand_touching(obj_idx)
        in_bowl = env.object_in_bowl(obj_idx)

        if not picked:
            if h > LIFT_HEIGHT_M and held:
                dwell_t0 = t if dwell_t0 is None else dwell_t0
                if t - dwell_t0 >= DWELL_S:
                    picked = True
            else:
                dwell_t0 = None

        if in_bowl and env.object_speed(obj_idx) < ARRIVAL_SPEED_M_S and not held:
            arrive_t0 = t if arrive_t0 is None else arrive_t0
            if t - arrive_t0 >= ARRIVAL_DWELL_S:
                return RolloutResult(True, "success", step + 1, t, max_h, picked,
                                     n_drops,
                                     np.array(obs_log) if record else None,
                                     np.array(act_log) if record else None)
        else:
            arrive_t0 = None

        if picked and h <= LIFT_HEIGHT_M and not held and not in_bowl:
            drop_t0 = t if drop_t0 is None else drop_t0
            if t - drop_t0 >= DROP_DWELL_S:
                n_drops += 1
                picked = False
                dwell_t0 = drop_t0 = None
        else:
            drop_t0 = None

    return RolloutResult(False, "timeout", max_steps, float(env.data.time), max_h,
                         picked, n_drops,
                         np.array(obs_log) if record else None,
                         np.array(act_log) if record else None)
