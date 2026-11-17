#!/usr/bin/env python3
"""Per-phase trajectory visualisation for the collected demonstrations.

Three views, because "visualise the trajectories per phase" means different
things depending on what you are asking:

  --view overlay   (default) Palm paths in 3D, one panel per phase, every
                   episode overlaid and coloured by condition. Answers "what
                   does this phase look like, and do the two conditions do it
                   differently?" Good for a talk.

  --view spread    Cross-episode spread within each phase, as a function of
                   progress through that phase. Answers "WHERE in the phase do
                   the demonstrations disagree?" This is the analytic view --
                   it is the per-phase version of the band plot that showed
                   contact-aware staying wide while baseline re-converges.

  --view budget    How long each phase takes, per episode, per condition.
                   Answers "where does the time go?" Contact-aware spends 60%
                   of its rows in APPROACH against baseline's 25%, which is a
                   finding in itself and invisible in the other two views.

Phases come from the `phase` array stored in each episode .npz, which is
reconstructed from the EVENT STREAM rather than read from the trace's own
`phase` column -- that column omits PICK for every baseline mode (see
logs.phase_per_row). Checked on data/v4_still: 0 of 97 episodes are missing
PICK.

Note these are TASK phases (APPROACH / PICK / TRANSPORT / PLACE), not the
controller phases (PLAN / REACH / GRASP) that plot_trajectories.py colours by.
The task phase says what the operator is doing; the controller phase says who
is driving. Both are worth looking at and they do not align.

PLACE is ~0% of rows -- it is a single terminal frame per episode -- so it is
dropped from the per-phase panels by default rather than shown as a dot.

Usage:

    python scripts/plot_phases.py --data data/v4_still
    python scripts/plot_phases.py --view spread --obj obj_036_wood_block
    python scripts/plot_phases.py --view budget
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from handil.data.schema import obs_slice

C = {"baseline": "#1F77B4", "contact_aware": "#D62728"}
LABEL = {"baseline": "baseline", "contact_aware": "contact-aware"}
PHASES = ("APPROACH", "PICK", "TRANSPORT", "PLACE")
GRID = 60          # resample points per phase segment


def load(data: Path, obj: str | None, retargeter: str):
    """Episodes grouped by pipeline, each with palm xyz, joints and phase."""
    man = json.loads((data / "manifest.json").read_text())
    palm = obs_slice("palm_pos")
    arm = obs_slice("q_arm")
    out: dict[str, list] = {"baseline": [], "contact_aware": []}
    for e in man["episodes"]:
        if e["retargeter"] != retargeter or e.get("ablation"):
            continue
        if e["outcome"] != "success" or e["pipeline"] not in out:
            continue
        if obj and e["obj"] != obj:
            continue
        with np.load(data / f"{e['episode_id']}.npz", allow_pickle=True) as z:
            obs, ph, t = z["obs"], z["phase"], z["t"]
        xyz = np.nan_to_num(obs[:, palm])
        if not np.isfinite(xyz).all() or len(xyz) < 20:
            continue
        out[e["pipeline"]].append(
            {"xyz": xyz, "arm": np.nan_to_num(obs[:, arm]),
             "phase": np.asarray(ph), "t": t - t[0], "obj": e["obj"]})
    return out


def segment(ep, phase):
    """Rows of one episode belonging to one phase, as a contiguous slice.

    Phases can in principle recur (a failed pick re-entering APPROACH), so this
    takes the LONGEST contiguous run rather than every matching row -- stitching
    disjoint runs together would draw a line through space the arm never
    travelled.
    """
    m = ep["phase"] == phase
    if not m.any():
        return None
    idx = np.flatnonzero(m)
    splits = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
    run = max(splits, key=len)
    return run if len(run) >= 5 else None


def resample(a, n=GRID):
    src = np.linspace(0, 1, len(a))
    g = np.linspace(0, 1, n)
    return np.stack([np.interp(g, src, a[:, i]) for i in range(a.shape[1])], 1)


# ------------------------------------------------------------------ views


def view_overlay(data, out, title_extra, show_place):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phases = PHASES if show_place else PHASES[:3]
    fig = plt.figure(figsize=(5.0 * len(phases), 5.2))
    for i, phase in enumerate(phases, 1):
        ax = fig.add_subplot(1, len(phases), i, projection="3d")
        n = {}
        for pipe, eps in data.items():
            k = 0
            for ep in eps:
                run = segment(ep, phase)
                if run is None:
                    continue
                p = ep["xyz"][run]
                ax.plot(p[:, 0], p[:, 1], p[:, 2], lw=0.9,
                        color=C[pipe], alpha=0.45)
                ax.scatter(*p[0], s=14, color=C[pipe], alpha=0.8)
                k += 1
            n[pipe] = k
        ax.set_title(f"{phase}\n" + "  ".join(
            f"{LABEL[p]} n={v}" for p, v in n.items()), fontsize=9)
        ax.set_xlabel("x (m)", fontsize=7)
        ax.set_ylabel("y (m)", fontsize=7)
        ax.set_zlabel("z (m)", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.suptitle("Palm trajectory by task phase — dot marks the phase start"
                 + title_extra, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=165)
    print(f"-> {out}")


def view_spread(data, out, title_extra, show_place):
    """Cross-episode spread within each phase, against progress through it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phases = PHASES if show_place else PHASES[:3]
    fig, axes = plt.subplots(1, len(phases), figsize=(4.6 * len(phases), 4.4),
                             sharey=True)
    g = np.linspace(0, 100, GRID)
    summary = {}
    for ax, phase in zip(np.atleast_1d(axes), phases):
        for pipe, eps in data.items():
            S = [resample(ep["xyz"][r]) for ep in eps
                 if (r := segment(ep, phase)) is not None]
            if len(S) < 3:
                continue
            A = np.array(S)
            w = np.linalg.norm(A.std(0), axis=1) * 100
            ax.plot(g, w, color=C[pipe], lw=2.2,
                    label=f"{LABEL[pipe]} (n={len(S)})")
            ax.fill_between(g, 0, w, color=C[pipe], alpha=0.12)
            summary.setdefault(phase, {})[pipe] = float(w.mean())
        ax.set_title(phase, fontsize=11)
        ax.set_xlabel("progress through phase (%)", fontsize=9)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel(
        "cross-episode palm spread (cm)", fontsize=9)
    fig.suptitle("Where WITHIN each phase the demonstrations disagree"
                 + title_extra, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=165)
    print(f"-> {out}")
    print("\n  mean spread (cm) per phase:")
    print(f"    {'phase':12s} {'baseline':>10s} {'contact_aw':>12s} {'ratio':>7s}")
    for ph in phases:
        s = summary.get(ph, {})
        b, c = s.get("baseline"), s.get("contact_aware")
        if b and c:
            print(f"    {ph:12s} {b:10.2f} {c:12.2f} {c/b:6.2f}x")


def view_budget(data, out, title_extra, show_place):
    """Time spent in each phase, per episode."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phases = PHASES if show_place else PHASES[:3]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # Left: absolute seconds per phase, box per condition.
    ax = axes[0]
    pos, ticks = [], []
    for i, phase in enumerate(phases):
        for j, pipe in enumerate(("baseline", "contact_aware")):
            vals = []
            for ep in data[pipe]:
                run = segment(ep, phase)
                if run is None:
                    continue
                vals.append(float(ep["t"][run[-1]] - ep["t"][run[0]]))
            if not vals:
                continue
            p = i * 3 + j
            bp = ax.boxplot([vals], positions=[p], widths=0.7,
                            patch_artist=True)
            bp["boxes"][0].set_facecolor(C[pipe])
            bp["boxes"][0].set_alpha(0.45)
            pos.append(p)
        ticks.append((i * 3 + 0.5, phase))
    ax.set_xticks([t[0] for t in ticks])
    ax.set_xticklabels([t[1] for t in ticks], fontsize=9)
    ax.set_ylabel("phase duration (s)", fontsize=9)
    ax.set_title("How long each phase takes", fontsize=10)
    ax.grid(alpha=0.25, axis="y")

    # Right: share of the episode, which is the comparable quantity when the
    # two conditions have different total durations.
    ax = axes[1]
    width = 0.36
    for j, pipe in enumerate(("baseline", "contact_aware")):
        means = []
        for phase in phases:
            fr = []
            for ep in data[pipe]:
                run = segment(ep, phase)
                fr.append(len(run) / len(ep["phase"]) if run is not None else 0.0)
            means.append(100 * np.mean(fr) if fr else 0.0)
        ax.bar(np.arange(len(phases)) + (j - .5) * width, means, width,
               color=C[pipe], alpha=0.85, label=LABEL[pipe])
    ax.set_xticks(np.arange(len(phases)))
    ax.set_xticklabels(phases, fontsize=9)
    ax.set_ylabel("share of episode rows (%)", fontsize=9)
    ax.set_title("Where the episode's time goes", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Phase budget" + title_extra, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=165)
    print(f"-> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/v4_still"))
    ap.add_argument("--view", default="overlay",
                    choices=["overlay", "spread", "budget"])
    ap.add_argument("--obj", default=None,
                    help="restrict to one object (default: pool all four)")
    ap.add_argument("--retargeter", default="anyteleop")
    ap.add_argument("--show-place", action="store_true",
                    help="include PLACE, which is ~1 row per episode")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    data = load(args.data, args.obj, args.retargeter)
    for pipe, eps in data.items():
        print(f"  {pipe:14s} {len(eps)} episodes")
    if min(len(v) for v in data.values()) < 3:
        raise SystemExit("too few episodes after filtering")

    extra = f"\n{args.data}" + (f" · {args.obj}" if args.obj else " · all objects")
    out = args.out or Path(f"docs/figs/phases_{args.view}"
                           + (f"_{args.obj}" if args.obj else "") + ".png")
    out.parent.mkdir(parents=True, exist_ok=True)

    {"overlay": view_overlay, "spread": view_spread,
     "budget": view_budget}[args.view](data, out, extra, args.show_place)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
