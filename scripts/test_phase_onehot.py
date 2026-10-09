"""A/B: does the phase one-hot change the contact-aware result?"""
import json
from pathlib import Path
import numpy as np
from handil.train import TrainConfig, train

rows = []
for data, tag in (("data/v2", "no_phase"), ("data/v2_ph", "phase_onehot")):
    for algo in ("bc_mlp", "bc_rnn"):
        for pipe in ("baseline", "contact_aware"):
            for seed in (0, 1, 2):
                cfg = TrainConfig(algo=algo, data_dir=data,
                    out_dir=f"runs/ab/{tag}_{algo}_{pipe}_s{seed}",
                    pipeline=pipe, retargeter="anyteleop",
                    exclude_ablations=True, max_episodes=16, epochs=60, seed=seed)
                s = train(cfg)
                rows.append(dict(obs=tag, algo=algo, pipeline=pipe, seed=seed,
                                 best_val=s["best_val"]))
                print(f"  {tag:12s} {algo:7s} {pipe:14s} s{seed} val={s['best_val']:.5f}",
                      flush=True)
Path("runs/ab/results.json").write_text(json.dumps(rows, indent=2))

print("\n=== phase one-hot A/B (3 seeds, 16 episodes/cell) ===")
print(f"{'obs':14s} {'algo':8s} {'baseline':>20s} {'contact_aware':>20s} {'ratio':>7s}")
for tag in ("no_phase", "phase_onehot"):
    for algo in ("bc_mlp", "bc_rnn"):
        g = {}
        for pipe in ("baseline", "contact_aware"):
            v = [r["best_val"] for r in rows
                 if r["obs"] == tag and r["algo"] == algo and r["pipeline"] == pipe]
            g[pipe] = (np.mean(v), np.std(v))
        b, c = g["baseline"], g["contact_aware"]
        print(f"{tag:14s} {algo:8s} {b[0]:9.5f}±{b[1]:.5f} {c[0]:11.5f}±{c[1]:.5f} "
              f"{b[0]/c[0]:6.2f}x")
