"""Turning hand-control runs into a learning-ready episode dataset.

One episode = one trial (a single pick-and-place attempt), sliced out of the
run's pose_trace by the trial's wall-clock window.

Why wall clock and not sim time: sim time resets to 0 on Backspace and on
MuJoCo BADQACC auto-resets, and real runs in the archive contain up to 4 such
resets. `t_wall` is monotonic. (The trial logger's own timeout is likewise
wall-clock.)

Output is one .npz per episode plus a manifest.json, so training can memory-map
shards and filter by method/outcome/object without re-reading the source logs.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import actions as act
from . import logs as L
from .methods import Method, resolve
from .schema import (ACT_DIM, N_ARM, N_ROBOT, OBJ_INDEX, OBJ_MASS, OBJ_SIZE,
                     OBS_FIELDS, obj_onehot, obs_dim, phase_onehot)

# Palm body index within the logged xpos/xquat arrays is scene-dependent, so
# the palm pose is resolved by name once per scene and cached here.
PALM_BODY = "leap_palm"

# Summed per-finger normal force above which the hand counts as touching the
# object. Well above solver noise, well below a real pinch (demos peak 50-170 N).
TOUCH_FORCE_N = 0.1


@dataclass
class EpisodeMeta:
    episode_id: str
    run: str
    trial_id: int
    method_key: str
    pipeline: str
    retargeter: str
    ablation: str | None
    method_conflict: bool
    obj: str
    # Index into the TRACE's obj_qpos spawn order -- NOT into
    # Scene.object_bodies. See _trace_object_index. Use `obj` (the name) for
    # anything scene-side.
    obj_idx: int
    outcome: str
    n_steps: int
    duration_s: float
    action_space: str           # "position" (rad) or "torque" (N*m)
    action_source: str          # how the labels were obtained
    kp_hand: float
    dt_median: float
    n_objects: int
    limit_viol_max_rad: float
    grasp_repair_frac: float    # rows whose finger targets came from qpos[t+1]
    dropped_rows: int = 0       # rows removed by drop_phases (0 when unused)


def _palm_pose(trace: dict, sl: slice, palm_bid: int | None,
               n: int) -> tuple[np.ndarray, np.ndarray]:
    """Palm position/orientation in world for rows `sl`, or NaN if unavailable."""
    if (palm_bid is not None and "xpos" in trace and "xquat" in trace
            and trace["xpos"].ndim == 3 and palm_bid < trace["xpos"].shape[1]):
        return (trace["xpos"][sl, palm_bid, :].copy(),
                trace["xquat"][sl, palm_bid, :].copy())
    return np.full((n, 3), np.nan), np.full((n, 4), np.nan)


def _object_in_palm(trace: dict, sl: slice, palm_bid: int | None, obj_idx: int,
                    n: int) -> tuple[np.ndarray, np.ndarray]:
    """Object pose in the palm frame, recomputed from logged world poses.

    The trace's own `obj_in_palm` column is unusable: the source only fills it
    for the *proximate* object during an active trial and leaves NaN otherwise,
    which is all-NaN in 41 of 44 ICRA episodes. Since `xpos`/`xquat` (palm) and
    `obj_qpos` (object freejoints) are both logged, the transform is recoverable
    for every row:

        p_palm_frame = R_palm^T (p_obj - p_palm)
        R_rel        = R_palm^T R_obj

    matching kinova_leap_pick_place.py:6014-6030. This is the most task-relevant
    signal the policy gets -- where the object sits relative to the hand -- so it
    is worth reconstructing rather than dropping.
    """
    out_p = np.full((n, 3), np.nan)
    out_q = np.full((n, 4), np.nan)
    if palm_bid is None or "xpos" not in trace or "xquat" not in trace:
        return out_p, out_q
    if "obj_qpos" not in trace or trace["xpos"].ndim != 3:
        return out_p, out_q
    oq = trace["obj_qpos"][sl]
    if oq.ndim != 2 or oq.shape[1] < 7 * (obj_idx + 1):
        return out_p, out_q

    p_palm = trace["xpos"][sl, palm_bid, :]
    q_palm = trace["xquat"][sl, palm_bid, :]
    p_obj = oq[:, 7 * obj_idx: 7 * obj_idx + 3]
    q_obj = oq[:, 7 * obj_idx + 3: 7 * obj_idx + 7]

    R_palm = _quat_to_mat(q_palm)                      # (n,3,3)
    d = (p_obj - p_palm)[:, :, None]
    out_p = np.einsum("nij,njk->nik", np.swapaxes(R_palm, 1, 2), d)[:, :, 0]
    R_rel = np.einsum("nij,njk->nik", np.swapaxes(R_palm, 1, 2),
                      _quat_to_mat(q_obj))
    out_q = _mat_to_quat(R_rel)
    return out_p, out_q


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Batched wxyz quaternion -> rotation matrix."""
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], 1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], 1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1),
    ], 1)


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    """Batched rotation matrix -> wxyz quaternion (Shepperd's method)."""
    n = R.shape[0]
    out = np.zeros((n, 4))
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    big = tr > 0
    s = np.sqrt(np.maximum(tr[big] + 1.0, 1e-12)) * 2
    out[big, 0] = 0.25 * s
    out[big, 1] = (R[big, 2, 1] - R[big, 1, 2]) / s
    out[big, 2] = (R[big, 0, 2] - R[big, 2, 0]) / s
    out[big, 3] = (R[big, 1, 0] - R[big, 0, 1]) / s
    for i in np.flatnonzero(~big):           # rare branch, per-row is fine
        m = R[i]
        k = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
        if k == 0:
            s = np.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12)) * 2
            out[i] = [(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                      (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif k == 1:
            s = np.sqrt(max(1.0 - m[0, 0] + m[1, 1] - m[2, 2], 1e-12)) * 2
            out[i] = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                      0.25 * s, (m[1, 2] + m[2, 1]) / s]
        else:
            s = np.sqrt(max(1.0 - m[0, 0] - m[1, 1] + m[2, 2], 1e-12)) * 2
            out[i] = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                      (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    return out


def _height_above_rest(trace: dict, sl: slice, obj_idx: int,
                       n: int) -> np.ndarray:
    """Object lift above its own resting height, metres.

    The rest height is taken as the object's own median z over the episode's
    first 10% of rows, when it is still sitting on the counter -- the same
    quantity `PickPlaceEnv._rest_z` captures at reset.
    """
    if "obj_qpos" not in trace:
        return np.zeros((n, 1))
    oq = trace["obj_qpos"][sl]
    if oq.ndim != 2 or oq.shape[1] < 7 * (obj_idx + 1):
        return np.zeros((n, 1))
    z = oq[:, 7 * obj_idx + 2]
    head = max(1, int(0.1 * len(z)))
    h = z - float(np.median(z[:head]))
    # The terminal PLACE frame is sampled AFTER the trial tears down and stows
    # the object at z = -5, which would put -5.9 m into this channel (measured:
    # 30 of 34938 rows) and wreck the normalizer's std. Stowed rows carry no
    # task information, so clamp to the plausible lift range.
    return np.clip(h, -0.05, 1.0)[:, None]


def build_obs(trace: dict, sl: slice, palm_bid: int | None,
              obj_idx: int = 0, obj_name: str = "",
              phase: np.ndarray | None = None,
              extras: tuple[str, ...] = ()) -> np.ndarray:
    """Pack the canonical observation for rows `sl` of a loaded trace."""
    qpos = trace["qpos"][sl]
    n = qpos.shape[0]
    q = qpos[:, :N_ROBOT]

    if "qvel" in trace:
        qd = trace["qvel"][sl][:, :N_ROBOT]
    else:
        qd = np.zeros((n, N_ROBOT))

    def get(key, width):
        if key in trace:
            a = np.atleast_2d(trace[key][sl])
            if a.shape[0] != n:
                a = a.reshape(n, -1)
            if a.shape[1] >= width:
                return a[:, :width].astype(float)
            out = np.zeros((n, width))
            out[:, : a.shape[1]] = a
            return out
        return np.full((n, width), np.nan)

    palm_pos, palm_quat = _palm_pose(trace, sl, palm_bid, n)

    # Prefer the recomputed palm-frame pose; fall back to the logged column only
    # where reconstruction is impossible (older traces lacking xpos/obj_qpos).
    oip, oiq = _object_in_palm(trace, sl, palm_bid, obj_idx, n)
    if np.isnan(oip).all():
        oip, oiq = get("obj_in_palm", 3), get("obj_quat_in_palm", 4)

    parts = {
        "q_arm": q[:, :N_ARM],
        "q_hand": q[:, N_ARM:N_ROBOT],
        "qd_arm": qd[:, :N_ARM],
        "qd_hand": qd[:, N_ARM:N_ROBOT],
        "obj_in_palm": oip,
        "obj_quat_in_palm": oiq,
        "palm_pos": palm_pos,
        "palm_quat": palm_quat,
        "norm_force": L.pad_forces(get("norm_force", 4)),
        "tan_force": L.pad_forces(get("tan_force", 4)),
        # Object conditioning: keyed on the NAME, so it cannot inherit the
        # trace-vs-scene index mismatch (see _trace_object_index).
        "obj_id_onehot": np.tile(obj_onehot(obj_name), (n, 1)),
        "obj_size": np.tile(OBJ_SIZE.get(obj_name, (0.0, 0.0, 0.0)), (n, 1)),
        "obj_mass": np.full((n, 1), OBJ_MASS.get(obj_name, 0.0)),
        # Task-stage proxy, computed from the logged object height and the
        # per-finger normal force -- both already in the trace, so the env can
        # reproduce them exactly from live state.
        "height_above_rest": _height_above_rest(trace, sl, obj_idx, n),
        "hand_touching": (L.pad_forces(get("norm_force", 4)).sum(1, keepdims=True)
                          > TOUCH_FORCE_N).astype(float),
    }
    blocks = [parts[name] for name, _ in OBS_FIELDS]
    if "phase_onehot" in extras:
        if phase is None or len(phase) != n:
            blocks.append(np.zeros((n, 4)))
        else:
            blocks.append(np.array([phase_onehot(p) for p in phase], dtype=float))
    obs = np.concatenate(blocks, axis=1)
    assert obs.shape == (n, obs_dim(extras)), obs.shape
    return obs


def _trial_rows(trace: dict, tr: L.TrialRecord) -> slice | None:
    """Row range for a trial, using monotonic wall clock."""
    if "t_wall" not in trace:
        return None
    tw = trace["t_wall"]
    if not np.isfinite(tr.t_start_wall) or not np.isfinite(tr.t_end_wall):
        return None
    i0, i1 = np.searchsorted(tw, [tr.t_start_wall, tr.t_end_wall])
    i0, i1 = int(i0), int(min(i1 + 1, len(tw)))
    return slice(i0, i1) if i1 - i0 >= 2 else None


def _trace_object_index(trace: dict, sl: slice, obj_name: str,
                        object_bodies: list[str] | None) -> int:
    """Index of the trial's object **in the log trace's own spawn order**.

    FRAME WARNING. The returned index addresses `obj_qpos` columns in the
    recorded trace, i.e. whatever order that session spawned objects in. It is
    NOT an index into `Scene.object_bodies`, and for the ICRA scene the two
    disagree on 3 of the 4 objects:

        object        trace idx   scene idx
        wood_block        0           0
        orange            1           2
        mug               2           3
        gelatin_box       3           1

    Passing this into `PickPlaceEnv(active_object=...)` therefore activates the
    WRONG object. Use the object NAME for anything scene-side; this index is only
    valid for slicing the trace it came from.

    Sequential-spawn runs stow every inactive object below the floor at z=-5, so
    the active one is simply the object that is above the floor -- more reliable
    than matching names against a body order that varies with the scene build.
    Falls back to the scene's body list, then to index 0.
    """
    if "obj_qpos" in trace:
        oq = trace["obj_qpos"][sl]
        if oq.ndim == 2 and oq.shape[1] >= 7:
            n_obj = oq.shape[1] // 7
            # "Live" = above the floor. Stowed objects are parked at z=-5 with
            # collision off, and since nothing holds them there they keep
            # free-falling -- real runs reach z=-360 -- so test against a
            # generous threshold, and count rows rather than taking a median
            # (a median over a falling object is meaningless).
            live_frac = np.array([
                float(np.mean(oq[:, 7 * i + 2] > -1.0)) for i in range(n_obj)])
            live = np.flatnonzero(live_frac > 0.5)
            if len(live) == 1:
                return int(live[0])
            if len(live) > 1 and object_bodies:
                # Several live objects: disambiguate by name.
                for i in live:
                    if i < len(object_bodies) and object_bodies[i] == obj_name:
                        return int(i)
            if len(live) >= 1:
                return int(live[np.argmax(live_frac[live])])
    if object_bodies and obj_name in object_bodies:
        return object_bodies.index(obj_name)
    return 0


def extract_run(run_dir: Path, out_dir: Path, kp_arm: np.ndarray,
                palm_bid: int | None, jnt_range: np.ndarray | None,
                jnt_limited: np.ndarray | None = None,
                action_space: str = "position",
                outcomes: tuple[str, ...] = ("success",),
                min_steps: int = 20,
                object_bodies: list[str] | None = None,
                icra_only: bool = True,
                nominal_only: bool = True,
                require_n_objects: int | None = 4,
                extras: tuple[str, ...] = (),
                drop_phases: tuple[str, ...] = (),
                collapse_still: float = 0.0,
                collapse_keep: int = 1) -> list[EpisodeMeta]:
    """Extract every qualifying trial in one run. Returns per-episode metadata.

    `drop_phases` removes rows whose `control_phase` is listed, for isolating
    what a phase contributes. The case this was written for is PLAN: in the
    October runs the straight line from the operator's hand-over pose to the
    pregrasp is almost always blocked (route=linear 79% -> 0%), so the RRT
    planner actually runs, and the arm HOLDS STILL for a median ~8 s while it
    does. That is ~850 near-zero-action frames per episode whose duration
    varies unpredictably -- a plausible aliasing source that has nothing to do
    with grasp certification per se. Dropping them separates "dead time hurts"
    from "the planned paths hurt".

    Rows are dropped BEFORE the action labels and observations are built, so
    the remaining rows keep their own exact labels; the episode simply has a
    discontinuity where the hold was. That is the right surgery for a
    single-step policy and WRONG for a chunked one (ACT would learn to step
    across the seam), so use it with bc_mlp/bc_rnn and read ACT results from an
    unfiltered dataset.

    `collapse_still` is the sequence-model-safe alternative, and the better
    default. Instead of deleting a phase wholesale it deletes only rows where
    the robot DID NOT MOVE -- max |dq| over the 23 joints below this many
    radians -- keeping the first `collapse_keep` frames of each stationary run
    so the pause is still represented, just not at its arbitrary length.
    Measured on the contact-aware episodes at a 1e-4 rad threshold:

        PLAN    94.6% of rows stationary     (the RRT-solving hold)
        REACH   13.8%                        (waypoint replay -- real motion)
        GRASP    1.0%                        (continuous closing)

    So it targets the hold almost exactly without naming a phase, and it
    leaves a continuous trajectory behind: every surviving transition is one
    the robot actually made, which is what a chunked or recurrent policy needs.
    A chunk spanning a collapsed pause sees the arm resume rather than teleport.
    """
    info = L.inspect_run(run_dir)
    if not info.usable or not info.trials:
        return []
    trace = L.load_trace(run_dir)
    events = L.read_events(run_dir / "events.jsonl")
    out: list[EpisodeMeta] = []

    for tr in info.trials:
        if outcomes and tr.outcome not in outcomes:
            continue
        # Nominal-only: the ablation sweep scales object mass (x1.25-2.5) and
        # friction, which build_scene() cannot reproduce, so including those
        # episodes would put an obj_mass channel in the observation that the
        # rollout env can never match.
        if icra_only and tr.obj not in OBJ_INDEX:
            continue
        if require_n_objects is not None and L.n_objects(info.nq) != require_n_objects:
            continue
        sl = _trial_rows(trace, tr)
        if sl is None or (sl.stop - sl.start) < min_steps:
            continue

        # Prefer the logged --mode over the directory name: the directory is a
        # human-chosen string and the per-trial method field drops the
        # retargeter. See logs.run_mode.
        _mode, _from_log = L.run_mode(events, info.mode)
        method: Method = resolve(_mode, tr.method)
        if nominal_only and method.ablation:
            continue
        kp_hand = act.kp_hand_for_mode(tr.method)
        # The event method is lossy about the retargeter but the finger gain
        # only depends on the pipeline, which the event gets right.

        qpos = trace["qpos"][sl]
        repair = 0.0
        if action_space == "torque":
            if not info.has_torque:
                continue        # cannot form a torque label without the fields
            a = act.torques_from_log(trace["qfrc_applied"][sl],
                                     trace["qfrc_bias"][sl])
            src = "logged_torque"
        elif info.has_torque:
            cp = trace["control_phase"][sl] if "control_phase" in trace else None
            if "q_grasp_hold" in trace and "kp_eff" in trace:
                # The run logged the GRASP hold directly, so those rows need no
                # inversion and no repair -- the commanded target IS the hold.
                a, exact = act.targets_from_logged_grasp(
                    qpos, trace["qfrc_applied"][sl], trace["qfrc_bias"][sl],
                    kp_arm, kp_hand, trace["q_grasp_hold"][sl],
                    trace["kp_eff"][sl], cp)
                src = "logged_grasp_hold" if exact > 0 else "inverted_torque"
                repair = 0.0
            else:
                a = act.targets_from_torque(
                    qpos, trace["qfrc_applied"][sl], trace["qfrc_bias"][sl],
                    kp_arm, kp_hand)
                # No logged hold: the contact-aware GRASP phase adds non-PD
                # torques, so those finger targets must be repaired instead.
                a, repair = act.blend_grasp_phase(a, qpos, cp)
                src = "inverted_torque"
        else:
            a = act.targets_from_next_qpos(qpos)
            src = "next_qpos"

        if action_space == "delta":
            # Residual on the current position; joint limits are enforced at
            # rollout after adding qpos back, not on the delta itself.
            a = act.deltas_from_targets(a, qpos)
            src += "_delta"
            rep = {"n": int(a.shape[0]), "min": float(a.min()),
                   "max": float(a.max())}
        elif action_space == "torque":
            # Torques are not joint positions; limits do not apply.
            rep = {"n": int(a.shape[0]), "min": float(a.min()),
                   "max": float(a.max())}
        else:
            rep = act.sanity_report(a, jnt_range, jnt_limited)  # BEFORE clipping
            if jnt_range is not None:
                a = act.clip_to_limits(a, jnt_range, jnt_limited)

        obj_idx = _trace_object_index(trace, sl, tr.obj, object_bodies)
        t = trace["t_wall"][sl]
        ph = L.phase_per_row(events, tr.trial_id, t)
        obs = build_obs(trace, sl, palm_bid, obj_idx=obj_idx, obj_name=tr.obj,
                        phase=ph, extras=extras)

        # Optional control-phase filter. Applied AFTER labels and observations
        # are built, so every surviving row keeps the label it would have had;
        # only whole rows disappear. `control_phase` is the raw controller
        # state (PLAN/REACH/GRASP), distinct from `ph` above, which is the
        # reconstructed task phase.
        n_dropped = 0
        keep = np.ones(obs.shape[0], dtype=bool)
        if drop_phases and "control_phase" in trace:
            cp = trace["control_phase"][sl]
            cp = np.array([c.decode() if isinstance(c, bytes) else str(c)
                           for c in cp])
            keep &= ~np.isin(cp, list(drop_phases))
        if collapse_still > 0:
            # Motion between consecutive logged frames. The first row has no
            # predecessor and is always kept.
            dq = np.abs(np.diff(qpos[:, :N_ROBOT], axis=0)).max(1)
            moving = np.concatenate([[True], dq >= collapse_still])
            # Keep the first `collapse_keep` frames of each stationary run, so
            # a pause survives as a pause rather than vanishing.
            still_keep = np.zeros_like(moving)
            run_len = 0
            for i, mv in enumerate(moving):
                if mv:
                    run_len = 0
                else:
                    run_len += 1
                    if run_len <= collapse_keep:
                        still_keep[i] = True
            keep &= (moving | still_keep)
        if not keep.all():
            n_dropped = int((~keep).sum())
            if keep.sum() < min_steps:
                continue
            obs, a, t = obs[keep], a[keep], t[keep]
            ph = ph[keep]
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0

        ep_id = f"{run_dir.name}__t{tr.trial_id:04d}"
        meta = EpisodeMeta(
            episode_id=ep_id, run=run_dir.name, trial_id=tr.trial_id,
            method_key=method.key, pipeline=method.pipeline,
            retargeter=method.retargeter, ablation=method.ablation,
            method_conflict=method.conflict, obj=tr.obj, obj_idx=int(obj_idx),
            outcome=tr.outcome,
            n_steps=int(obs.shape[0]),
            duration_s=float(t[-1] - t[0]),
            action_space=action_space,
            action_source=src, kp_hand=float(kp_hand), dt_median=dt,
            n_objects=L.n_objects(info.nq),
            limit_viol_max_rad=float(rep.get("limit_viol_max_rad", np.nan)),
            grasp_repair_frac=float(repair),
            dropped_rows=int(n_dropped),
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / f"{ep_id}.npz",
            obs=obs.astype(np.float32), action=a.astype(np.float32),
            t=t.astype(np.float64),
            # Reconstructed from the event stream, NOT the trace's own `phase`
            # column: that column omits PICK for every baseline mode. See
            # logs.phase_per_row.
            phase=ph,
            meta=json.dumps(asdict(meta)))
        out.append(meta)
    return out
