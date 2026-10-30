#!/usr/bin/env python3
"""Visualise WHERE the two pipelines' demonstrations disagree.

The headline numbers (action divergence 1.74-1.86x worse, per-block validation
loss 7.5-7.9x worse on the arm) say contact-aware demonstrations are harder to
learn from, but a ratio does not show a reviewer *what* is different. These
four panels do.

The measurement behind every panel: resample each successful episode to 100
normalised-time points (so a 16 s and a 60 s trial are comparable), then take
the cross-episode spread at each point. That spread -- not the mean path -- is
the quantity the policy has to deal with: at a given stage of the task, how
much do the demonstrations disagree about where the palm should be?

Panel A  tube plot: mean palm path with a +-1 sd band, one row per condition.
         The band IS the disagreement; a fat band means a policy conditioned on
         observation alone cannot know which path it is on.
Panel B  the band width against normalised time, both conditions overlaid.
         This is the panel that carries the argument: both start tight (~5 cm),
         baseline re-converges by the end (~9 cm), contact-aware stays wide
         (13-19 cm). Lock-in times are rugged underneath, so the divergence can
         be read against when the planner took over.
Panel C  per-episode path length. Contact-aware paths are SHORTER and far more
         uniform in length (sd 0.29 m vs 0.85 m) -- the certificate does make
         the motion tidier by this measure, which is why "more variance" needs
         to be stated carefully.
Panel D  arm vs hand joint spread, the decomposition that matches the per-block
         loss result.

Written to be read alongside the loss numbers, not instead of them: it is the
same claim in a form that survives a projector.

Usage:  python scripts/plot_variance.py --obj obj_036_wood_block
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

C_BASE = "#1F77B4"
C_CA = "#D62728"
GRID = 100          # normalised-time samples per episode

_BID: list = []


def palm_bid():
    if not _BID:
        from handil.envs.scene import build_scene
        _BID.append(int(build_scene("robocasa").palm_bid))
    return _BID[0]


def episode(run: str, tid: int):
    """Palm xyz, arm joints, hand joints, lock-in fraction -- all on [0,1]."""
    d = LOGS / run
    if not (d / "pose_trace.npz").exists() or not (d / "events.jsonl").exists():
        return None
    try:
        tr = load_trace(d)
    except Exception:
        return None
    evs = read_events(d / "events.jsonl")
    rec = {t.trial_id: t for t in trials_from_events(evs)}
    t = rec.get(tid)
    if t is None or "xpos" not in tr or tr["xpos"].ndim != 3:
        return None
    tw = tr["t_wall"]
    m = (tw >= t.t_start_wall) & (tw <= t.t_end_wall)
    if m.sum() < 30:
        return None

    xyz = tr["xpos"][m, palm_bid(), :]
    q = tr["q_robot"][m]
    src = np.linspace(0, 1, len(xyz))
    g = np.linspace(0, 1, GRID)
    rs = lambda a: np.stack([np.interp(g, src, a[:, i]) for i in range(a.shape[1])], 1)

    # Lock-in only exists for contact-aware (rec_accept). Baseline has NO
    # hand-over: pick_confirmed is an outcome, not a decision, so it is not a
    # comparable event and is deliberately not collected here.
    lock = None
    t0, t1 = tw[m][0], tw[m][-1]
    for e in evs:
        if e.get("trial_id") == tid and e.get("event") == "rec_accept":
            lock = (float(e["t_wall"]) - t0) / max(t1 - t0, 1e-9)
            break
    raw_len = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
    return {"xyz": rs(xyz), "arm": rs(q[:, :7]), "hand": rs(q[:, 7:23]),
            "lock": lock, "path_len": raw_len}


def gather(obj: str, pipe: str, retargeter="anyteleop"):
    man = json.loads((DATA / "manifest.json").read_text())
    eps = [e for e in man["episodes"]
           if e["pipeline"] == pipe and e["retargeter"] == retargeter
           and not e.get("ablation") and e["outcome"] == "success"
           and e["obj"] == obj]
    out = []
    for e in eps:
        r = episode(e["run"], e["trial_id"])
        if r is not None:
            out.append(r)
    return out


def band(eps, key="xyz"):
    """Cross-episode mean and spread at each normalised-time point."""
    A = np.array([e[key] for e in eps])              # (N, GRID, D)
    return A.mean(0), A.std(0), A


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obj", default="obj_036_wood_block")
    ap.add_argument("--out", type=Path, default=Path("docs/figs/variance.png"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {p: gather(args.obj, p) for p in ("baseline", "contact_aware")}
    for p, v in data.items():
        print(f"  {p:14s} {len(v)} episodes")
        if len(v) < 3:
            raise SystemExit(f"too few episodes for {p}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14, 9))
    g = np.linspace(0, 100, GRID)

    # --- A: tube plots, one per condition -------------------------------
    for j, (pipe, col) in enumerate((("baseline", C_BASE),
                                     ("contact_aware", C_CA))):
        eps = data[pipe]
        mu, sd, A = band(eps)
        ax = fig.add_subplot(2, 3, 1 + j, projection="3d")
        for a in A:
            ax.plot(a[:, 0], a[:, 1], a[:, 2], lw=0.7, color=col, alpha=0.35)
        ax.plot(mu[:, 0], mu[:, 1], mu[:, 2], lw=2.4, color="k", label="mean")
        # +-1 sd as a scatter cloud at every 5th point, sized by local spread
        k = slice(None, None, 5)
        s = np.linalg.norm(sd[k], axis=1)
        ax.scatter(mu[k, 0], mu[k, 1], mu[k, 2], s=40 + 2400 * s,
                   color=col, alpha=0.18, edgecolors="none")
        ax.set_title(f"{pipe}  (n={len(eps)})\nmean path, bubble = ±1 sd",
                     fontsize=9)
        ax.set_xlabel("x (m)", fontsize=7)
        ax.set_ylabel("y (m)", fontsize=7)
        ax.set_zlabel("z (m)", fontsize=7)
        ax.tick_params(labelsize=6)

    # --- B: band width vs normalised time -------------------------------
    ax = fig.add_subplot(2, 3, 3)
    for pipe, col in (("baseline", C_BASE), ("contact_aware", C_CA)):
        _mu, sd, _A = band(data[pipe])
        w = np.linalg.norm(sd, axis=1) * 100
        ax.plot(g, w, color=col, lw=2.2, label=f"{pipe} (n={len(data[pipe])})")
        ax.fill_between(g, 0, w, color=col, alpha=0.12)
    locks = [e["lock"] for e in data["contact_aware"] if e["lock"] is not None]
    for i, L in enumerate(locks):
        ax.axvline(L * 100, color=C_CA, ls=":", lw=1.0, alpha=0.75,
                   label="contact-aware lock-in" if i == 0 else None)
    ax.set_xlabel("normalised time through trial (%)", fontsize=8)
    ax.set_ylabel("cross-episode palm spread (cm)", fontsize=8)
    ax.set_title("Where the demonstrations disagree", fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)

    # --- C: path length ---------------------------------------------------
    ax = fig.add_subplot(2, 3, 4)
    vals = [[e["path_len"] for e in data[p]] for p in ("baseline", "contact_aware")]
    bp = ax.boxplot(vals, tick_labels=["baseline", "contact-aware"],
                    patch_artist=True, widths=0.5)
    for patch, col in zip(bp["boxes"], (C_BASE, C_CA)):
        patch.set_facecolor(col)
        patch.set_alpha(0.35)
    for i, v in enumerate(vals, 1):
        ax.scatter(np.full(len(v), i) + np.random.uniform(-.07, .07, len(v)),
                   v, s=22, color="k", zorder=3, alpha=0.7)
    ax.set_ylabel("palm path length (m)", fontsize=8)
    ax.set_title("Contact-aware paths are SHORTER\nand more uniform in length",
                 fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, axis="y")

    # --- D: arm vs hand joint spread -------------------------------------
    ax = fig.add_subplot(2, 3, 5)
    width = 0.35
    for i, (pipe, col) in enumerate((("baseline", C_BASE),
                                     ("contact_aware", C_CA))):
        _m, sd_a, _ = band(data[pipe], "arm")
        _m, sd_h, _ = band(data[pipe], "hand")
        ax.bar([0 + (i - .5) * width, 1 + (i - .5) * width],
               [sd_a.mean(), sd_h.mean()], width, color=col, alpha=0.8,
               label=pipe)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["arm (7 joints)", "hand (16 joints)"], fontsize=8)
    ax.set_ylabel("cross-episode joint spread (rad)", fontsize=8)
    ax.set_title("The disagreement is in the ARM", fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, axis="y")

    # --- E: the numbers, as text -----------------------------------------
    ax = fig.add_subplot(2, 3, 6)
    ax.axis("off")
    rows = []
    for pipe in ("baseline", "contact_aware"):
        _m, sd, _ = band(data[pipe])
        w = np.linalg.norm(sd, axis=1) * 100
        rows.append((pipe, w[:10].mean(), w[-10:].mean(), w.max()))
    txt = ["Palm spread (cm), cross-episode", ""]
    txt.append(f"{'':16s}{'start':>8s}{'end':>8s}{'peak':>8s}")
    for p, a, b_, c in rows:
        txt.append(f"{p:16s}{a:8.1f}{b_:8.1f}{c:8.1f}")
    txt += ["", "Both start tight. Baseline re-converges;",
            "contact-aware stays wide to the end.", "",
            "Validation loss, contact-aware / baseline:",
            "   arm   7.9x (BC-MLP)   7.5x (BC-RNN)",
            "   hand  2.6x (BC-MLP)   3.2x (BC-RNN)", "",
            f"RRT-driven share of contact-aware frames:",
            "   59% mean  (range 24-86%)",
            "Baseline has no planner, and no hand-over."]
    ax.text(0.0, 1.0, "\n".join(txt), va="top", ha="left", fontsize=8.5,
            family="monospace", transform=ax.transAxes)

    fig.suptitle(
        f"Where contact-aware demonstrations disagree — {args.obj}\n"
        "every episode resampled to 100 normalised-time points, "
        "spread taken across episodes", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=165)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
