#!/usr/bin/env python3
"""Plot the palm trajectory of training episodes in 3D Cartesian space.

Three questions this answers, all raised about what the policy actually learns
from:

  1. **Is the RRT solution in the training data?**  Yes. The contact-aware
     pipeline plans with RRT after lock-in and replays the waypoints
     kinematically (control_phase PLAN -> REACH), and `extract.py` never
     filters on control_phase -- it uses it only to choose a label-recovery
     method. So planner-driven motion is ~59% of a contact-aware episode's
     frames (median 55%, range 24-86%), and 0% of a baseline episode's.
     Colouring by control_phase makes that visible rather than asserted.

  2. **What does a failure look like?**  `data/v3` kept successes only
     (`outcomes_kept: ["success"]`), so a failed episode has to be read from
     the raw log. Plotting one alongside a success shows whether failures
     differ in path or only in the final grasp.

  3. **Where does time go?**  Points are sampled at a fixed stride and sized
     by time, so a cluster of markers means the arm dwelt there. Dwell is
     invisible in a plain line plot and matters here: the lock-in wait is
     dead time in the middle of the trajectory.

Cartesian rather than joint space on purpose -- 7 joint curves are unreadable,
and the quantity the policy must get right is where the palm goes.

Usage:

    python scripts/plot_trajectories.py --obj obj_036_wood_block
    python scripts/plot_trajectories.py --overlay --n 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from handil.data.logs import load_trace, read_events, trials_from_events

LOGS = Path("/home/aipexws6/sureshkj/hand-control/logs")
DATA = Path("/home/aipexws6/sureshkj/hand-control-il/data/v3")

# control_phase initialises to 'REACH' at kinova_leap_pick_place.py:5514 and is
# only reassigned inside contact-aware branches, so in a BASELINE run every row
# reads 'REACH' and means "teleop". Phase colouring is therefore only
# meaningful for contact-aware runs.
PHASE_COLOR = {"PLAN": "#D62728", "REACH": "#1F77B4", "GRASP": "#2CA02C"}
PHASE_LABEL = {"PLAN": "PLAN (RRT solving)",
               "REACH": "REACH (RRT waypoint replay)",
               "GRASP": "GRASP (certified squeeze)"}


_PALM_BID: list = []


def _palm_bid():
    """Body index of leap_palm, resolved once from the scene."""
    if not _PALM_BID:
        try:
            from handil.envs.scene import build_scene
            _PALM_BID.append(int(build_scene("robocasa").palm_bid))
        except Exception:
            _PALM_BID.append(None)
    return _PALM_BID[0]


def _phases(tr, m):
    cp = tr.get("control_phase")
    if cp is None:
        return None
    cp = cp[m]
    return np.array([c.decode() if isinstance(c, bytes) else str(c) for c in cp])


def episode_xyz(run: str, trial_id: int):
    """Palm XYZ, time, and control phase for one trial, trimmed to the trial."""
    d = LOGS / run
    # Not every archived run kept its pose trace, and some predate the
    # palm_pos column; skip rather than fail so the caller can try the next.
    if not (d / "pose_trace.npz").exists() or not (d / "events.jsonl").exists():
        return None
    try:
        tr = load_trace(d)
    except Exception:
        return None
    evs = read_events(d / "events.jsonl")
    rec = {t.trial_id: t for t in trials_from_events(evs)}
    t = rec.get(trial_id)
    if t is None:
        return None
    tw = tr["t_wall"]
    m = (tw >= t.t_start_wall) & (tw <= t.t_end_wall)
    if m.sum() < 10:
        return None
    # The trace has no palm_pos column; palm position is xpos indexed by the
    # leap_palm body, exactly as extract.py:_palm_pose resolves it.
    if "xpos" not in tr or tr["xpos"].ndim != 3:
        return None
    bid = _palm_bid()
    if bid is None or bid >= tr["xpos"].shape[1]:
        return None
    xyz = tr["xpos"][m, bid, :]
    rel = tw[m] - tw[m][0]
    ph = _phases(tr, m)
    # Lock-in, for a marker: rec_accept (contact-aware) else pick_confirmed.
    lock = None
    for want in ("rec_accept", "pick_confirmed"):
        for e in evs:
            if e.get("trial_id") == trial_id and e.get("event") == want:
                lock = float(e["t_wall"]) - tw[m][0]
                break
        if lock is not None:
            break
    return {"xyz": xyz, "t": rel, "phase": ph, "lock": lock,
            "outcome": t.outcome, "obj": t.obj, "run": run, "trial": trial_id}


def find_failure(pipeline: str, obj: str | None, retargeter: str = "anyteleop"):
    """A failed/abandoned trial from the raw logs (data/v3 kept successes only)."""
    pat = ("contact_aware_w_anyteleop" if pipeline == "contact_aware"
           else "anyteleop")
    for d in sorted(LOGS.glob(f"{pat}*")):
        f = d / "events.jsonl"
        if not f.exists():
            continue
        try:
            evs = read_events(f)
        except Exception:
            continue
        for t in trials_from_events(evs):
            if t.outcome == "success":
                continue
            if obj and t.obj != obj:
                continue
            ep = episode_xyz(d.name, t.trial_id)
            if ep is not None:
                return ep
    return None


def _draw(ax, ep, stride, show_phase, label=None, color=None, alpha=1.0):
    xyz, t, ph = ep["xyz"], ep["t"], ep["phase"]
    k = slice(None, None, stride)
    x, y, z = xyz[k, 0], xyz[k, 1], xyz[k, 2]
    tt = t[k]
    # Size grows with time: a dense cluster of large markers = the arm dwelt.
    size = 6 + 26 * (tt / max(tt[-1], 1e-9))
    if show_phase and ph is not None:
        pk = ph[k]
        for name, col in PHASE_COLOR.items():
            sel = pk == name
            if sel.any():
                ax.scatter(x[sel], y[sel], z[sel], s=size[sel], c=col,
                           label=PHASE_LABEL[name], depthshade=False,
                           edgecolors="none", alpha=0.9)
        ax.plot(x, y, z, lw=0.6, color="0.6", alpha=0.7, zorder=0)
    else:
        ax.plot(x, y, z, lw=1.3, color=color, alpha=alpha, label=label)
    ax.scatter(*xyz[0], s=70, marker="o", facecolor="none",
               edgecolor="k", lw=1.4, zorder=5)
    ax.scatter(*xyz[-1], s=90, marker="X", c="k", zorder=5)
    if ep["lock"] is not None:
        i = int(np.argmin(np.abs(t - ep["lock"])))
        ax.scatter(*xyz[i], s=150, marker="*", c="#FF7F0E",
                   edgecolor="k", lw=0.6, zorder=6,
                   label=f"lock-in ({100*ep['lock']/t[-1]:.0f}%)")


def _tidy(ax, title):
    ax.set_xlabel("x (m)", fontsize=8)
    ax.set_ylabel("y (m)", fontsize=8)
    ax.set_zlabel("z (m)", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc="upper left")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obj", default="obj_036_wood_block")
    ap.add_argument("--stride", type=int, default=12,
                    help="sample every Nth frame as a point")
    ap.add_argument("--overlay", action="store_true",
                    help="overlay N episodes per condition instead of one each")
    ap.add_argument("--n", type=int, default=6, help="episodes per condition")
    ap.add_argument("--out", type=Path,
                    default=Path("docs/figs/trajectories.png"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    man = json.loads((DATA / "manifest.json").read_text())
    eps = [e for e in man["episodes"]
           if e["retargeter"] == "anyteleop" and not e.get("ablation")
           and e["outcome"] == "success" and e["obj"] == args.obj]

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.overlay:
        fig = plt.figure(figsize=(13, 6))
        for j, pipe in enumerate(("baseline", "contact_aware")):
            ax = fig.add_subplot(1, 2, j + 1, projection="3d")
            sel = [e for e in eps if e["pipeline"] == pipe][: args.n]
            cmap = plt.cm.viridis(np.linspace(0, 0.9, max(len(sel), 1)))
            n = 0
            for i, e in enumerate(sel):
                ep = episode_xyz(e["run"], e["trial_id"])
                if ep is None:
                    continue
                _draw(ax, ep, args.stride, False,
                      label=None, color=cmap[i], alpha=0.85)
                n += 1
            _tidy(ax, f"{pipe}  —  {n} successful episodes\n{args.obj}")
            # Legend would just list colours; replace with a count.
            ax.legend_.remove() if ax.legend_ else None
        fig.suptitle("Palm trajectories, overlaid  "
                     "(circle = start, X = end, star = lock-in)", fontsize=10)
        fig.tight_layout()
        fig.savefig(args.out, dpi=170)
        print(f"-> {args.out}")
        return 0

    # One success + one failure per condition, phase-coloured.
    fig = plt.figure(figsize=(13, 11))
    panels = []
    for pipe in ("baseline", "contact_aware"):
        sel = [e for e in eps if e["pipeline"] == pipe]
        ok = None
        for e in sel:
            ok = episode_xyz(e["run"], e["trial_id"])
            if ok is not None:
                break
        bad = find_failure(pipe, args.obj)
        panels += [(f"{pipe} — SUCCESS", ok, pipe),
                   (f"{pipe} — FAILURE", bad, pipe)]

    for i, (title, ep, pipe) in enumerate(panels, 1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        if ep is None:
            ax.text2D(0.5, 0.5, "no episode found", ha="center",
                      transform=ax.transAxes, fontsize=9)
            ax.set_title(title, fontsize=9)
            continue
        show = pipe == "contact_aware"
        _draw(ax, ep, args.stride, show, label="teleop", color="#1F77B4")
        dur = ep["t"][-1]
        _tidy(ax, f"{title}  ({ep['outcome']})\n{ep['run']} trial {ep['trial']}"
                  f"  ·  {dur:.1f}s  ·  marker size ∝ time")
        panels[i - 1] = (title, ep, pipe)

    fig.suptitle(
        "Palm trajectory in Cartesian space — circle = start, X = end, "
        "star = lock-in\nContact-aware is phase-coloured: red/blue frames are "
        "RRT-driven and ARE in the training labels", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=170)
    print(f"-> {args.out}")

    for title, ep, _ in panels:
        if ep is None:
            continue
        ph = ep["phase"]
        extra = ""
        # Only meaningful for contact-aware: in a baseline run control_phase
        # never leaves its 'REACH' default, which there means teleop.
        if ph is not None and "contact_aware" in title:
            rrt = np.isin(ph, ("PLAN", "REACH")).mean()
            extra = f"  RRT-driven {100*rrt:.0f}%"
        lk = f"  lock@{100*ep['lock']/ep['t'][-1]:.0f}%" if ep["lock"] else ""
        print(f"  {title:28s} {ep['run'][:40]:40s} t{ep['trial']}  "
              f"{ep['t'][-1]:5.1f}s{lk}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
