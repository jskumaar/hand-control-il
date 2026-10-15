"""Powered re-test of the contact-aware hypothesis, per-episode split.

At 3 seeds the per-episode BC-MLP effect (d = 0.81) needs 24 seeds per arm for
80% power. BC-MLP trains in ~5 s on GPU, so run them.
"""
import json
from pathlib import Path
import numpy as np
from handil.train import TrainConfig, train

N_SEEDS = 24
rows = []
for pipe in ("baseline", "contact_aware"):
    for seed in range(N_SEEDS):
        cfg = TrainConfig(
            algo="bc_mlp", data_dir="data/v2",
            out_dir=f"runs/power/{pipe}_s{seed}",
            pipeline=pipe, retargeter="anyteleop", exclude_ablations=True,
            max_episodes=16, epochs=60, seed=seed)
        s = train(cfg)
        rows.append(dict(pipeline=pipe, seed=seed, best_val=s["best_val"]))
        print(f"  {pipe:14s} s{seed:02d} val={s['best_val']:.5f}", flush=True)
        Path("runs/power/results.json").write_text(json.dumps(rows, indent=2))

b = np.array([r["best_val"] for r in rows if r["pipeline"] == "baseline"])
c = np.array([r["best_val"] for r in rows if r["pipeline"] == "contact_aware"])
sp = np.sqrt((b.std(ddof=1) ** 2 + c.std(ddof=1) ** 2) / 2)
d = (b.mean() - c.mean()) / sp
# Welch t, two-sided, normal approximation for the p-value
se = np.sqrt(b.var(ddof=1) / len(b) + c.var(ddof=1) / len(c))
t = (b.mean() - c.mean()) / se
from math import erfc, sqrt
pval = erfc(abs(t) / sqrt(2))

print("\n=== POWERED TEST, per-episode split, BC-MLP, %d seeds/arm ===" % N_SEEDS)
print(f"  baseline       {b.mean():.5f} +- {b.std(ddof=1):.5f}")
print(f"  contact_aware  {c.mean():.5f} +- {c.std(ddof=1):.5f}")
print(f"  effect {b.mean()-c.mean():+.5f}  Cohen d {d:.2f}  Welch t {t:.2f}  p ~ {pval:.4f}")
print(f"  verdict: {'contact_aware BETTER (significant)' if pval < 0.05 and d > 0 else 'NOT significant'}")
