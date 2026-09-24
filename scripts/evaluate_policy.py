#!/usr/bin/env python3
"""Score a trained policy by closed-loop rollout in the MuJoCo scene.

    python scripts/evaluate_policy.py --run runs/ca_any_act --episodes 20
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from handil.data.actions import kp_hand_for_mode
from handil.eval import evaluate
from handil.train import load_policy


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="a training run dir")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--kp-hand", type=float, default=None,
                    help="finger gain; default: inferred from the trained pipeline")
    ap.add_argument("--scene", default="pick_place")
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    policy = load_policy(args.run, device=args.device)

    kp_hand = args.kp_hand
    if kp_hand is None:
        cfg = json.loads((args.run / "config.json").read_text())
        # The finger gain follows the pipeline the demos came from, so a policy
        # trained on contact-aware data is rolled out with that pipeline's gain.
        kp_hand = 0.8 if cfg.get("pipeline") == "contact_aware" else 2.0
    print(f"rolling out {args.run.name}: {args.episodes} episodes, kp_hand={kp_hand}")

    summary, rows = evaluate(policy, n_episodes=args.episodes,
                             max_steps=args.max_steps, kp_hand=kp_hand,
                             scene_name=args.scene, root=args.root,
                             seed=args.seed, verbose=not args.quiet)

    out = {"summary": asdict(summary), "episodes": rows}
    (args.run / "eval.json").write_text(json.dumps(out, indent=2))
    print(f"\nsuccess {summary.success_rate:.1%}  pick {summary.pick_rate:.1%}  "
          f"drops/ep {summary.mean_drops:.2f}")
    print(f"outcomes: {summary.outcomes}")
    print(f"-> {args.run / 'eval.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
