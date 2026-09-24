#!/usr/bin/env python3
"""Replay recorded demonstrations through the rollout env.

Calibration, not evaluation: it measures the ceiling a learned policy could
reach in this environment, and tells you whether a low policy score means a
weak policy or a broken env.

    python scripts/replay_demos.py --data data/icra --limit 20
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from handil.envs.rollout import PickPlaceEnv
from handil.envs.scene import build_scene
from handil.eval import replay_episode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/icra"))
    ap.add_argument("--scene", default="robocasa")
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--method", help="only this method key")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    eps = json.loads((args.data / "manifest.json").read_text())["episodes"]
    if args.method:
        eps = [e for e in eps if e["method_key"] == args.method]
    if args.limit:
        eps = eps[: args.limit]
    if not eps:
        raise SystemExit("no episodes selected")

    scene = build_scene(args.scene, root=args.root)
    rows, by_method = [], {}
    print(f"replaying {len(eps)} demonstrations\n")
    for e in eps:
        d = np.load(args.data / f"{e['episode_id']}.npz", allow_pickle=True)
        env = PickPlaceEnv(scene=scene, kp_hand=e["kp_hand"],
                           active_object=e["obj_idx"])
        r = replay_episode(env, d["action"], obs_ref=d["obs"], t=d["t"],
                           episode_id=e["episode_id"], obj=e["obj"])
        rows.append({**asdict(r), "method_key": e["method_key"]})
        by_method.setdefault(e["method_key"], []).append(r)
        print(f"  {r.obj:<22} {'SUCCESS' if r.success else 'pick' if r.picked else '-':<8}"
              f" max_h={r.max_height:+.3f} arm_err_p95={r.arm_err_p95:.3f}"
              f" min|oip|={r.min_obj_in_palm:.3f}")

    print("\n" + "=" * 66)
    print(f"{'method':32s} {'n':>3s} {'pick':>7s} {'success':>8s}")
    print("-" * 66)
    for k, v in sorted(by_method.items()):
        print(f"{k:32s} {len(v):3d} "
              f"{sum(x.picked for x in v)/len(v):6.0%} "
              f"{sum(x.success for x in v)/len(v):7.0%}")
    n = len(rows)
    print(f"\noverall: pick {sum(r['picked'] for r in rows)/n:.0%}  "
          f"success {sum(r['success'] for r in rows)/n:.0%}  (n={n})")

    if args.out:
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
