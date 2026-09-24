#!/usr/bin/env python3
"""The headline experiment: collection method x algorithm.

Trains one policy per (method, algorithm) cell and scores each closed-loop,
then prints a success-rate table. This is the thing the repo exists to answer.

    python scripts/run_comparison.py --algos bc_mlp bc_rnn act --episodes 20
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from handil.algos import available
from handil.data.dataset import load_manifest
from handil.eval import evaluate
from handil.train import TrainConfig, load_policy, train


def method_cells(data_dir: Path, min_episodes: int,
                 pool_ablations: bool) -> list[tuple[str, dict]]:
    """Collection methods with enough successful episodes to train on."""
    eps = load_manifest(data_dir)
    key = "base_key" if pool_ablations else "method_key"
    counts: Counter = Counter()
    for e in eps:
        if e["outcome"] != "success":
            continue
        if pool_ablations:
            counts[f"{e['pipeline']}+{e['retargeter']}"] += 1
        else:
            if e.get("ablation"):
                continue
            counts[e["method_key"]] += 1
    out = []
    for k, n in counts.most_common():
        if n < min_episodes:
            continue
        pipeline, retargeter = k.split("+", 1)
        out.append((k, {"pipeline": pipeline, "retargeter": retargeter,
                        "n_episodes": n}))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/episodes"))
    ap.add_argument("--out", type=Path, default=Path("runs/comparison"))
    ap.add_argument("--algos", nargs="+", default=["bc_mlp", "bc_rnn"],
                    choices=available())
    ap.add_argument("--methods", nargs="*",
                    help="explicit method keys; default: every method with enough data")
    ap.add_argument("--min-episodes", type=int, default=10)
    ap.add_argument("--pool-ablations", action="store_true",
                    help="merge the solref sweeps into their base method")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--episodes", type=int, default=20, help="eval rollouts per cell")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    cells = method_cells(args.data, args.min_episodes, args.pool_ablations)
    if args.methods:
        want = set(args.methods)
        cells = [(k, v) for k, v in cells if k in want]
    if not cells:
        raise SystemExit("no collection method has enough episodes")

    print(f"{len(cells)} methods x {len(args.algos)} algorithms "
          f"= {len(cells)*len(args.algos)} cells")
    for k, v in cells:
        print(f"  {k:36s} {v['n_episodes']:4d} episodes")

    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for mkey, sel in cells:
        for algo in args.algos:
            tag = f"{mkey.replace('+','_')}__{algo}"
            run_dir = args.out / tag
            print(f"\n=== {mkey} | {algo} ===")
            cfg = TrainConfig(
                algo=algo, data_dir=str(args.data), out_dir=str(run_dir),
                pipeline=sel["pipeline"], retargeter=sel["retargeter"],
                exclude_ablations=not args.pool_ablations,
                epochs=args.epochs, seed=args.seed, device=args.device)
            s = train(cfg)
            row = {"method": mkey, "algo": algo,
                   "n_episodes": s["n_episodes"], "n_samples": s["n_samples"],
                   "best_val": s["best_val"], "train_s": s["train_s"]}

            if not args.skip_eval:
                kp = 0.8 if sel["pipeline"] == "contact_aware" else 2.0
                policy = load_policy(run_dir, device="cpu")
                summ, rows = evaluate(policy, n_episodes=args.episodes,
                                      max_steps=args.max_steps, kp_hand=kp,
                                      seed=args.seed, verbose=False)
                (run_dir / "eval.json").write_text(
                    json.dumps({"summary": asdict(summ), "episodes": rows}, indent=2))
                row.update(success_rate=summ.success_rate,
                           pick_rate=summ.pick_rate,
                           mean_max_height=summ.mean_max_height)
                print(f"  -> success {summ.success_rate:.1%}  "
                      f"pick {summ.pick_rate:.1%}")
            results.append(row)
            (args.out / "results.json").write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 78)
    print(f"{'collection method':30s} {'algo':9s} {'eps':>5s} {'val':>9s} "
          f"{'pick':>7s} {'success':>8s}")
    print("-" * 78)
    for r in results:
        sr = f"{r['success_rate']:.1%}" if "success_rate" in r else "-"
        pk = f"{r['pick_rate']:.1%}" if "pick_rate" in r else "-"
        print(f"{r['method']:30s} {r['algo']:9s} {r['n_episodes']:5d} "
              f"{r['best_val']:9.5f} {pk:>7s} {sr:>8s}")
    print(f"\n-> {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
