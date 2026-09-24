"""Rebuilding the hand-control MuJoCo scene in this repo.

Mirrors the source loader's spec surgery (kinova_leap_pick_place.py:672-814)
closely enough that a policy trained on the logs can be rolled out against the
same physics. The pieces that matter and are easy to get wrong:

  * The model is built from an MjSpec, edited, then compiled -- not loaded
    straight from XML. `base_link` is re-posed onto the table and yawed +90
    degrees about world z.
  * Hand self-collision is disabled by setting every `leap_*` geom's contype
    to 2, so the pair rule fails hand-hand but keeps hand-object.
  * Objects come from models/scene_objects.json as YCB meshes; the four
    primitive obj_* bodies in scene_pick_place.xml are DELETED in that path.
  * Object z is not in the JSON -- it is derived post-compile by dropping each
    object onto the table surface.

The source repo is imported by path rather than vendored, so the scene and the
YCB assets stay single-sourced. Set HANDCONTROL_ROOT or pass `root=`.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TABLE_TOP_Z = {"pick_place": 0.625, "robocasa": 0.86}
SCENE_FILES = {"pick_place": "models/scene_pick_place.xml",
               "robocasa": "models/scene_robocasa.xml"}

N_ROBOT = 23
# Palm-down teleop home (kinova_leap_pick_place.py:3568).
HOME_WRIST_DOWN = np.array([0.0045, 0.6033, 3.1416, -2.3985, 0.0048, 1.431, 3.1407])


def handcontrol_root(root: str | Path | None = None) -> Path:
    """Locate the source repo."""
    if root is not None:
        return Path(root).expanduser().resolve()
    env = os.environ.get("HANDCONTROL_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    guess = here.parents[3] / "hand-control"
    if guess.exists():
        return guess
    raise FileNotFoundError(
        "hand-control repo not found; set HANDCONTROL_ROOT or pass root=")


def full_mass_matrix(model, data) -> np.ndarray:
    """Dense joint-space inertia, across MuJoCo versions.

    `mj_fullM`'s signature changed between the 3.6 the source repo pins and
    3.14: it took the sparse `MjData.qM` array, and now takes the `MjData`
    itself (with `qM` renamed to `M`). Try the new form, fall back to the old,
    so this repo works against either.
    """
    import mujoco as mj

    out = np.zeros((model.nv, model.nv))
    try:
        mj.mj_fullM(model, data, out)          # MuJoCo >= ~3.9
    except TypeError:
        mj.mj_fullM(model, out, data.qM)       # MuJoCo <= ~3.8
    return out


@dataclass
class Scene:
    model: object          # mujoco.MjModel
    data: object           # mujoco.MjData
    palm_bid: int
    object_bodies: list[str]
    object_qadr: list[int]
    kp_arm: np.ndarray
    table_top_z: float
    # Collision masks exactly as compiled. Sequential-spawn stowing zeroes these
    # in place, so the pristine values must be captured at build time -- reading
    # them off a model an env has already stowed into returns zeros.
    contype0: np.ndarray = None
    conaff0: np.ndarray = None

    @property
    def n_objects(self) -> int:
        return len(self.object_bodies)


# The scene the ICRA benchmark demonstrations were actually recorded in.
#
# Recovered from the runs' own logged obj_qpos, not from models/scene_objects.json
# (which holds only the current one-gelatin-box config). Every ICRA run is:
#
#   * the RoboCasa counter scene, counter top at z = 0.86 -- NOT the pick_place
#     table at 0.625;
#   * four YCB objects with THREE stowed below the floor at z = -5, i.e.
#     --sequential-spawn: exactly one object is on the counter at a time;
#   * that one object presented at [-0.35, 0.35], the same point regardless of
#     which object it is, so initial conditions are identical across objects
#     and across runs. NOTE this is NOT the source's SEQ_SPAWN_XY constant
#     ([-0.20, 0.45]): the ICRA runs predate or override it. The value here is
#     measured from the runs themselves -- the active object's world position at
#     trial_start, median over all 52 ICRA trials, is [-0.350, 0.350, 0.860] for
#     every one of the four objects, with zero spread on three of them.
#   * presentation order counterbalanced by a cyclic Latin square, which
#     matters for the human study but not for a policy.
#
ICRA_OBJECTS = ("036_wood_block", "009_gelatin_box", "017_orange", "025_mug")
SEQ_SPAWN_XY = (-0.35, 0.35)
STOW_Z = -5.0


def icra_layout(objects: tuple[str, ...] = ICRA_OBJECTS) -> list[dict]:
    """Object config for the ICRA RoboCasa scene.

    All four are attached at the shared spawn point; `PickPlaceEnv` then stows
    every one but the active object, mirroring the source's sequential spawn.
    """
    return [{"id": oid, "xy": list(SEQ_SPAWN_XY)} for oid in objects]


def build_scene(scene: str = "robocasa", root: str | Path | None = None,
                objects: list[dict] | None = None,
                icra: bool = True) -> Scene:
    """Compile the scene the way the source repo does.

    Defaults to the ICRA benchmark configuration (RoboCasa counter, the four
    YCB objects), since that is what the demonstrations were recorded in.
    Pass `icra=False` to use the source repo's current scene_objects.json, or
    `objects=` to specify a layout outright.
    """
    if objects is None and icra:
        objects = icra_layout()
    import mujoco as mj

    rt = handcontrol_root(root)
    if str(rt) not in sys.path:
        sys.path.insert(0, str(rt))
    from environments import scene_objects as so  # noqa: E402

    top_z = TABLE_TOP_Z[scene]
    cwd = Path.cwd()
    os.chdir(rt)  # relative mesh/include paths in the XML resolve against the repo
    try:
        spec = mj.MjSpec.from_file(SCENE_FILES[scene])

        cfg = objects if objects is not None else so.load_scene_objects(scene)
        placed: list[tuple[str, list[float]]] = []
        if cfg:
            for k in list(spec.keys):          # keyframes are single-object
                spec.delete(k)
            for b in [b for b in spec.bodies if b.name.startswith("obj_")]:
                spec.delete(b)
            for e in cfg:
                xy = e.get("xy", [0.5, 0.3])
                bn = so.attach_ycb_object(
                    spec, e["id"], xy, mass=e.get("mass"),
                    friction=tuple(e["friction"]) if "friction" in e else (2.0, 0.05, 0.005),
                    quat=tuple(e["quat"]) if "quat" in e else (1.0, 0.0, 0.0, 0.0),
                    solref=tuple(e["solref"]) if "solref" in e else (0.004, 1.0))
                placed.append((bn, xy))

        s = 0.7071067811865476
        for b in spec.bodies:
            if b.name == "base_link":
                b.pos = [0.0, -0.15, top_z]
                b.quat = [s, 0.0, 0.0, s]
                break

        model = spec.compile()
        data = mj.MjData(model)

        for bn, _ in placed:
            so.place_on_surface(model, data, bn, top_z)
    finally:
        os.chdir(cwd)

    for j in (0, 2, 4, 6):                      # continuous joints
        model.jnt_range[j] = [-np.pi, np.pi]

    for gi in range(model.ngeom):
        bid = model.geom_bodyid[gi]
        bname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid) or ""
        if bname.startswith("leap_") and model.geom_contype[gi] != 0:
            model.geom_contype[gi] = 2          # disable hand self-collision

    # Arm gains + critical damping, at the wrist-down home (:3626-3635).
    data.qpos[:7] = HOME_WRIST_DOWN
    mj.mj_forward(model, data)
    inertia = np.clip(np.diag(full_mass_matrix(model, data))[:7], 1e-3, None)
    kp_arm = inertia * 30.0**2
    model.dof_damping[:7] = 2.0 * inertia * 30.0

    palm_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "leap_palm")

    obj_bodies, obj_qadr = [], []
    for bid in range(model.nbody):
        name = mj.mj_name2id and mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid) or ""
        if not name.startswith("obj_"):
            continue
        jadr = model.body_jntadr[bid]
        if jadr >= 0 and model.jnt_type[jadr] == mj.mjtJoint.mjJNT_FREE:
            obj_bodies.append(name)
            obj_qadr.append(int(model.jnt_qposadr[jadr]))

    return Scene(model=model, data=data, palm_bid=int(palm_bid),
                 object_bodies=obj_bodies, object_qadr=obj_qadr,
                 kp_arm=kp_arm, table_top_z=top_z,
                 contype0=model.geom_contype.copy(),
                 conaff0=model.geom_conaffinity.copy())


def home_qpos() -> np.ndarray:
    """Full 23-DOF home: wrist-down arm, middle/ring curled out of the way."""
    q = np.zeros(N_ROBOT)
    q[:7] = HOME_WRIST_DOWN
    q[11:15] = [1.2, 0.0, 0.5, 0.5]   # leap_mf_*
    q[15:19] = [1.2, 0.0, 0.5, 0.5]   # leap_rf_*
    return q
