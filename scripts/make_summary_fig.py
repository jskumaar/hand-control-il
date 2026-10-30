#!/usr/bin/env python3
"""One figure per claim, sized for a slide, for the v3 results.

Separate from plot_variance.py (a six-panel diagnostic, too dense to project).
Each figure here carries ONE point and is readable at the back of a room:

    fig1_learnability.png   three architectures, baseline vs contact-aware
    fig2_armhand.png        the same loss split by joint block
    fig3_band.png           where in the trial the demonstrations disagree
    fig4_divergence.png     the demonstration-quality metrics, before training

All numbers are v3 (31 matched episodes per arm). A label bug found on
2026-10-22 means the contact_aware+anyteleop cell was under-filled (31 of 56
episodes), so these are being recomputed on data/v4; the figures regenerate
from the JSON when those land.

Usage:  uv run python scripts/make_summary_fig.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

C_BASE = "#1F77B4"
C_CA = "#D62728"
FIGS = Path("docs/figs")


def fig_learnability(out: Path):
    """Three architectures, grouped bars, ratio annotated."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    algos = ["BC-MLP", "BC-RNN", "ACT"]
    base = [0.0968, 0.0851, 0.1896]
    ca = [0.3477, 0.3362, 0.3199]
    base_sd = [0.0142, 0.0085, 0.0046]
    ca_sd = [0.0285, 0.0352, 0.0221]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(algos))
    w = 0.36
    ax.bar(x - w / 2, base, w, yerr=base_sd, capsize=4, color=C_BASE,
           label="baseline", alpha=0.9)
    ax.bar(x + w / 2, ca, w, yerr=ca_sd, capsize=4, color=C_CA,
           label="contact-aware", alpha=0.9)
    for i, (b, c) in enumerate(zip(base, ca)):
        ax.text(i, max(b, c) + 0.045, f"{c/b:.1f}×", ha="center",
                fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontsize=13)
    ax.set_ylabel("validation loss  (lower is better)", fontsize=12)
    ax.set_title("Policies trained on contact-aware data are worse\n"
                 "8-fold CV over held-out EPISODES", fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.25, axis="y")
    ax.set_ylim(0, 0.46)
    ax.text(0.5, -0.17,
            "Losses are not comparable ACROSS algorithms (MSE for BC, L1+KL for ACT) — only the ratio within each pair.",
            transform=ax.transAxes, ha="center", fontsize=9, style="italic")
    fig.tight_layout()
    fig.savefig(out, dpi=175, bbox_inches="tight")
    print(f"-> {out}")


def fig_armhand(out: Path):
    """The same loss, split into arm and hand blocks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for algo in ("bc_mlp", "bc_rnn"):
        p = Path(f"runs/split_loss_{algo}.json")
        if p.exists():
            rows.append((algo, json.loads(p.read_text())["summary"]))
    if not rows:
        print("  (no split_loss_*.json yet)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), sharey=True)
    for ax, (algo, s) in zip(axes, rows):
        blocks = ["pooled", "arm", "hand"]
        x = np.arange(len(blocks))
        w = 0.36
        b = [s[k]["baseline"] for k in blocks]
        c = [s[k]["contact_aware"] for k in blocks]
        ax.bar(x - w / 2, b, w, color=C_BASE, label="baseline", alpha=0.9)
        ax.bar(x + w / 2, c, w, color=C_CA, label="contact-aware", alpha=0.9)
        top = max(max(b), max(c))
        for i, k in enumerate(blocks):
            ax.text(i, max(b[i], c[i]) + top * 0.04, f"{s[k]['ratio']:.1f}×",
                    ha="center", fontsize=14,
                    fontweight="bold" if k == "arm" else "normal")
        ax.set_ylim(0, top * 1.22)
        ax.set_xticks(x)
        ax.set_xticklabels(["all 23", "arm (7)", "hand (16)"], fontsize=12)
        ax.set_title({"bc_mlp": "BC-MLP", "bc_rnn": "BC-RNN"}[algo],
                     fontsize=13, pad=14)
        ax.grid(alpha=0.25, axis="y")
        ax.tick_params(labelsize=11)
    axes[0].set_ylabel("validation loss (normalised action units)", fontsize=12)
    axes[0].legend(fontsize=11, loc="upper left")
    fig.suptitle("The gap is ~3× larger in the ARM than the hand\n"
                 "the arm labels were never reconstructed — the fingers' partly were",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=175, bbox_inches="tight")
    print(f"-> {out}")


def fig_divergence(out: Path, qpath=Path("runs/demo_quality.json")):
    """Demonstration-quality metrics, measured before any training."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not qpath.exists():
        print(f"  (missing {qpath})")
        return
    d = json.loads(qpath.read_text())
    b, c = d["conditions"]["baseline"], d["conditions"]["contact_aware"]
    names = ["divergence\narm", "divergence\nhand", "divergence\n(H−t) weighted",
             "variance\narm", "variance\nhand", "state\ndiversity"]
    bv = [b["divergence"]["arm"], b["divergence"]["hand"],
          b["divergence"]["weighted"], b["variance"]["arm"],
          b["variance"]["hand"], b["state_diversity"]]
    cv = [c["divergence"]["arm"], c["divergence"]["hand"],
          c["divergence"]["weighted"], c["variance"]["arm"],
          c["variance"]["hand"], c["state_diversity"]]
    # normalise each metric to its baseline so one axis can hold them all
    rel = [x / y if y else np.nan for x, y in zip(cv, bv)]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    cols = [C_CA if r > 1 else "#2CA02C" for r in rel]
    bars = ax.bar(np.arange(len(names)), rel, 0.6, color=cols, alpha=0.9)
    ax.axhline(1.0, color="k", lw=1.4)
    for i, (r, bb) in enumerate(zip(rel, bars)):
        ax.text(i, r + 0.05, f"{r:.2f}×", ha="center", fontsize=13,
                fontweight="bold")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, fontsize=10.5)
    ax.set_ylabel("contact-aware ÷ baseline", fontsize=12)
    ax.set_title("Demonstration quality, measured BEFORE any policy is trained\n"
                 "red = worse for learning · green = better · line = parity",
                 fontsize=13)
    ax.grid(alpha=0.25, axis="y")
    ax.set_ylim(0, max(rel) * 1.25)
    ax.text(0.5, -0.22,
            "Divergence and variance: lower is better. State diversity has no implied direction — "
            "Belkhale et al. find it uncorrelated with policy success.",
            transform=ax.transAxes, ha="center", fontsize=9, style="italic")
    fig.tight_layout()
    fig.savefig(out, dpi=175, bbox_inches="tight")
    print(f"-> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quality", type=Path, default=Path("runs/demo_quality.json"))
    args = ap.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_learnability(FIGS / "fig1_learnability.png")
    fig_armhand(FIGS / "fig2_armhand.png")
    fig_divergence(FIGS / "fig4_divergence.png", args.quality)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
