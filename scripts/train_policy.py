#!/usr/bin/env python3
"""Train one policy on one slice of the demonstration data.

    python scripts/train_policy.py --algo act --pipeline contact_aware \
        --retargeter anyteleop --epochs 100 --out runs/ca_any_act
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from handil.algos import available
from handil.train import TrainConfig, train


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", default="bc_mlp", choices=available())
    ap.add_argument("--data", type=Path, default=Path("data/episodes"))
    ap.add_argument("--out", type=Path, default=Path("runs/default"))
    # Data selection -- the experimental variable.
    ap.add_argument("--pipeline", choices=["baseline", "contact_aware", "autonomous"])
    ap.add_argument("--retargeter", choices=["dexpilot", "anyteleop", "vwj",
                                             "vwj_upstream", "none"])
    ap.add_argument("--method-key", help="exact method key, e.g. contact_aware+vwj")
    ap.add_argument("--obj", help="restrict to one object body name")
    ap.add_argument("--outcomes", nargs="*", default=["success"])
    ap.add_argument("--include-ablations", action="store_true",
                    help="pool the solref sweep runs (@s0p4/6/8) with the rest")
    ap.add_argument("--max-episodes", type=int,
                    help="deterministic subsample, for data-scaling sweeps")
    # Optimisation.
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=None)
    args = ap.parse_args()

    algo_kwargs = {}
    if args.horizon is not None:
        algo_kwargs["horizon"] = args.horizon
    if args.chunk is not None:
        algo_kwargs["chunk"] = args.chunk

    cfg = TrainConfig(
        algo=args.algo, data_dir=str(args.data), out_dir=str(args.out),
        pipeline=args.pipeline, retargeter=args.retargeter,
        method_key=args.method_key, obj=args.obj,
        outcome=tuple(args.outcomes),
        exclude_ablations=not args.include_ablations,
        max_episodes=args.max_episodes,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        seed=args.seed, device=args.device, algo_kwargs=algo_kwargs)

    print(f"algo={cfg.algo}")
    print(f"selection: pipeline={cfg.pipeline} retargeter={cfg.retargeter} "
          f"method_key={cfg.method_key} outcomes={cfg.outcome}")
    s = train(cfg)
    print(f"\n{s['n_episodes']} episodes, {s['n_samples']} samples, "
          f"best val {s['best_val']:.5f} in {s['train_s']:.0f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
