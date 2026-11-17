#!/usr/bin/env python3
"""Per-object, per-phase trajectory grid: 3D paths and the same paths vs time.

A grid of objects x phases, with baseline and contact-aware overlaid in every
cell, so the question "does the pipeline change what this phase looks like, and
does the answer depend on the object?" is answerable by reading across a row.

Two layouts:

  --mode 3d     (default) palm path in Cartesian space, one small 3D axes per
                cell. Shows the SHAPE of the motion.

  --mode time   each palm axis against progress through the phase, stacked
                x/y/z within the cell. Shows WHEN things diverge, which the 3D
                view hides -- two paths can trace the same curve at different
                speeds and look identical in 3D.

Both normalise time to [0,1] WITHIN the phase, because the phases differ in
duration by condition (contact-aware spends 53% of its episode in APPROACH
against baseline's 28%) and comparing raw seconds would conflate "takes longer"
with "goes somewhere different". Use --view budget in plot_phases.py for the
duration question; this script is about the path.

Grouping is by object rather than start location on purpose: measured on
data/v4_still, object start positions have std 2.7-7.4 cm, which is spawn
jitter plus session drift rather than a designed factor, and binning on it
would make cells of 2-3 episodes. Object is the factor that was actually
controlled -- 8-17 episodes per (object, condition) cell.

PLACE is excluded by default (~1 row per episode).

Usage:

    python scripts/plot_phase_grid.py                      # 3D grid
    python scripts/plot_phase_grid.py --mode time          # vs-time grid
    python scripts/plot_phase_grid.py --mode time --axis z # one axis, bigger
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
PHASES = ("APPROACH", "PICK", "TRANSPORT")
OBJECTS = ("obj_036_wood_block", "obj_009_gelatin_box",
           "obj_017_orange", "obj_025_mug")
SHORT = {"obj_036_wood_block": "wood block", "obj_009_gelatin_box": "gelatin box",
         "obj_017_orange": "orange", "obj_025_mug": "mug"}
GRID = 50


def segment(phase_arr, phase):
    """Longest contiguous run of `phase`, or None if too short.

    Longest run rather than every matching row: a failed pick can re-enter
    APPROACH, and stitching disjoint runs would draw a line through space the
    arm never travelled.
    """
    m = phase_arr == phase
    if not m.any():
        return None
    idx = np.flatnonzero(m)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
    run = max(runs, key=len)
    return run if len(run) >= 5 else None


def resample(a, n=GRID):
    src = np.linspace(0, 1, len(a))
    g = np.linspace(0, 1, n)
    return np.stack([np.interp(g, src, a[:, i]) for i in range(a.shape[1])], 1)


def load(data: Path, retargeter: str):
    """{(object, pipeline): [ {xyz, phase} ]}"""
    man = json.loads((data / "manifest.json").read_text())
    pp = obs_slice("palm_pos")
    out: dict[tuple, list] = {}
    for e in man["episodes"]:
        if (e["retargeter"] != retargeter or e.get("ablation")
                or e["outcome"] != "success"
                or e["pipeline"] not in C or e["obj"] not in OBJECTS):
            continue
        with np.load(data / f"{e['episode_id']}.npz", allow_pickle=True) as z:
            obs, ph = z["obs"], np.asarray(z["phase"])
        xyz = np.nan_to_num(obs[:, pp])
        if not np.isfinite(xyz).all() or len(xyz) < 20:
            continue
        out.setdefault((e["obj"], e["pipeline"]), []).append(
            {"xyz": xyz, "phase": ph})
    return out


def cell_curves(data, obj, pipe, phase):
    """Resampled palm paths for one (object, condition, phase) cell."""
    out = []
    for ep in data.get((obj, pipe), []):
        run = segment(ep["phase"], phase)
        if run is not None:
            out.append(resample(ep["xyz"][run]))
    return np.array(out) if out else None


def plot_3d(data, out, data_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(4.2 * len(PHASES), 3.9 * len(OBJECTS)))
    for r, obj in enumerate(OBJECTS):
        for c, phase in enumerate(PHASES):
            ax = fig.add_subplot(len(OBJECTS), len(PHASES),
                                 r * len(PHASES) + c + 1, projection="3d")
            counts = {}
            for pipe in ("baseline", "contact_aware"):
                A = cell_curves(data, obj, pipe, phase)
                if A is None:
                    counts[pipe] = 0
                    continue
                counts[pipe] = len(A)
                for a in A:
                    ax.plot(a[:, 0], a[:, 1], a[:, 2], lw=0.8,
                            color=C[pipe], alpha=0.40)
                mu = A.mean(0)
                ax.plot(mu[:, 0], mu[:, 1], mu[:, 2], lw=2.4, color=C[pipe])
                ax.scatter(*mu[0], s=38, color=C[pipe],
                           edgecolor="k", lw=0.5, zorder=5)
            ax.set_title(
                f"{SHORT[obj]} · {phase}\n"
                + "  ".join(f"{LABEL[p]} n={counts.get(p,0)}" for p in C),
                fontsize=8)
            ax.tick_params(labelsize=5.5)
            ax.set_xlabel("x", fontsize=6)
            ax.set_ylabel("y", fontsize=6)
            ax.set_zlabel("z", fontsize=6)
    fig.suptitle(
        "Palm path by object and task phase — thin = episodes, thick = mean, "
        "dot = phase start\n"
        f"{data_name} · time normalised within each phase", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


def plot_time(data, out, data_name, axis):
    """Palm coordinate(s) against progress through the phase."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axes_idx = {"x": [0], "y": [1], "z": [2], "all": [0, 1, 2]}[axis]
    names = ["x", "y", "z"]
    nrow = len(OBJECTS) * len(axes_idx)
    fig, axs = plt.subplots(nrow, len(PHASES),
                            figsize=(4.4 * len(PHASES), 2.3 * nrow),
                            squeeze=False)
    g = np.linspace(0, 100, GRID)
    for r, obj in enumerate(OBJECTS):
        for k, ai in enumerate(axes_idx):
            for c, phase in enumerate(PHASES):
                ax = axs[r * len(axes_idx) + k][c]
                for pipe in ("baseline", "contact_aware"):
                    A = cell_curves(data, obj, pipe, phase)
                    if A is None:
                        continue
                    for a in A:
                        ax.plot(g, a[:, ai], lw=0.6, color=C[pipe], alpha=0.30)
                    mu, sd = A[:, :, ai].mean(0), A[:, :, ai].std(0)
                    ax.plot(g, mu, lw=2.2, color=C[pipe],
                            label=f"{LABEL[pipe]} (n={len(A)})")
                    ax.fill_between(g, mu - sd, mu + sd,
                                    color=C[pipe], alpha=0.15)
                ax.grid(alpha=0.22)
                ax.tick_params(labelsize=7)
                if r == 0 and k == 0:
                    ax.set_title(phase, fontsize=11)
                if c == 0:
                    ax.set_ylabel(f"{SHORT[obj]}\npalm {names[ai]} (m)",
                                  fontsize=8)
                if r == 0 and k == 0 and c == 0:
                    ax.legend(fontsize=7)
                if r == len(OBJECTS) - 1 and k == len(axes_idx) - 1:
                    ax.set_xlabel("progress through phase (%)", fontsize=8)
    fig.suptitle(
        "Palm position against progress through each phase — "
        "thin = episodes, thick = mean, band = ±1 sd\n"
        f"{data_name} · time normalised WITHIN the phase, so this shows path "
        "shape, not duration", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/v4_still"))
    ap.add_argument("--mode", default="3d", choices=["3d", "time"])
    ap.add_argument("--axis", default="all", choices=["x", "y", "z", "all"],
                    help="time mode only: which palm axis to draw")
    ap.add_argument("--retargeter", default="anyteleop")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    data = load(args.data, args.retargeter)
    print("  cell counts (object x condition):")
    for obj in OBJECTS:
        row = "  ".join(f"{LABEL[p]}={len(data.get((obj,p),[])):2d}" for p in C)
        print(f"    {SHORT[obj]:14s} {row}")

    suffix = f"_{args.axis}" if args.mode == "time" and args.axis != "all" else ""
    out = args.out or Path(f"docs/figs/phase_grid_{args.mode}{suffix}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "3d":
        plot_3d(data, out, str(args.data))
    else:
        plot_time(data, out, str(args.data), args.axis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
