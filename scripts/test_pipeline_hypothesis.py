"""Controlled test of the contact-aware hypothesis.

Pipeline is the ONLY variable: same retargeter (anyteleop), same object set,
matched episode counts, and 3 seeds per cell so a difference can be told from
seed noise.
"""
import json, sys
from pathlib import Path
import numpy as np
from handil.train import TrainConfig, train

import os
DATA = os.environ.get("HANDIL_DATA", "data/icra_all")
N_EPS = 60          # both cells have >= 63
SEEDS = (0, 1, 2)
ALGOS = ("bc_mlp", "bc_rnn")

rows = []
for algo in ALGOS:
    for pipe in ("baseline", "contact_aware"):
        for seed in SEEDS:
            cfg = TrainConfig(
                algo=algo, data_dir=DATA,
                out_dir=f"runs/{os.environ.get('HANDIL_TAG','pipe')}/{algo}_{pipe}_s{seed}",
                pipeline=pipe, retargeter="anyteleop",
                exclude_ablations=False, max_episodes=N_EPS,
                epochs=60, seed=seed)
            s = train(cfg)
            rows.append(dict(algo=algo, pipeline=pipe, seed=seed,
                             n_episodes=s["n_episodes"], n_samples=s["n_samples"],
                             best_val=s["best_val"]))
            print(f"  {algo:8s} {pipe:14s} seed{seed} "
                  f"eps={s['n_episodes']:3d} samples={s['n_samples']:6d} "
                  f"val={s['best_val']:.5f}", flush=True)

Path(f"runs/{os.environ.get('HANDIL_TAG','pipe')}/results.json").write_text(json.dumps(rows, indent=2))
print("\n=== pipeline effect, same retargeter (anyteleop), %d episodes ===" % N_EPS)
for algo in ALGOS:
    out = {}
    for pipe in ("baseline", "contact_aware"):
        v = [r["best_val"] for r in rows if r["algo"] == algo and r["pipeline"] == pipe]
        ns = [r["n_samples"] for r in rows if r["algo"] == algo and r["pipeline"] == pipe]
        out[pipe] = (np.mean(v), np.std(v), int(np.mean(ns)))
    b, c = out["baseline"], out["contact_aware"]
    better = "contact_aware" if c[0] < b[0] else "baseline"
    print(f"{algo:8s} baseline {b[0]:.5f}+-{b[1]:.5f} (n={b[2]})  "
          f"contact_aware {c[0]:.5f}+-{c[1]:.5f} (n={c[2]})  -> {better}")
