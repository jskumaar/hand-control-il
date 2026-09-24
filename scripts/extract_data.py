#!/usr/bin/env python3
"""Build a learning-ready episode dataset from hand-control run logs.

    python scripts/extract_data.py --logs ~/sureshkj/hand-control/logs \
        --out data/episodes --outcomes success
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from handil.data import logs as L
from handil.data.extract import extract_run
from handil.envs.scene import build_scene


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", type=Path, required=True, help="hand-control logs/ dir")
    ap.add_argument("--out", type=Path, default=Path("data/episodes"))
    ap.add_argument("--root", type=Path, default=None,
                    help="hand-control repo root (default: env or sibling dir)")
    ap.add_argument("--scene", default="robocasa", choices=["robocasa", "pick_place"],
                    help="scene the runs were recorded in (ICRA runs are robocasa)")
    ap.add_argument("--outcomes", nargs="*", default=["success"],
                    choices=["success", "timeout", "abandoned"],
                    help="which trial outcomes to keep (default: success only)")
    ap.add_argument("--include-obsolete", action="store_true")
    ap.add_argument("--min-steps", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None, help="first N runs only")
    args = ap.parse_args()

    print(f"scanning {args.logs} ...")
    runs = L.find_runs(args.logs, include_obsolete=args.include_obsolete)
    if args.limit:
        runs = runs[: args.limit]
    print(f"  {len(runs)} run directories")

    print("building scene for gains / palm index ...")
    scene = build_scene(args.scene, root=args.root)
    jnt_range = np.asarray(scene.model.jnt_range)
    print(f"  kp_arm = {np.array2string(scene.kp_arm, precision=1)}")

    # The palm body index within the logged xpos array is the compiled body id,
    # which depends on the object count of the run's own scene. Runs whose nq
    # differs from ours get NaN palm pose rather than a wrong body.
    our_nq = scene.model.nq

    args.out.mkdir(parents=True, exist_ok=True)
    metas, skipped = [], Counter()
    for i, run in enumerate(runs):
        try:
            info = L.inspect_run(run)
        except Exception as e:                      # corrupt npz
            skipped[f"unreadable: {type(e).__name__}"] += 1
            continue
        if not info.usable:
            skipped["unusable"] += 1
            continue
        if not info.trials:
            skipped["no trials"] += 1
            continue
        palm = scene.palm_bid if info.nq == our_nq else None
        try:
            got = extract_run(run, args.out, scene.kp_arm, palm, jnt_range,
                              outcomes=tuple(args.outcomes),
                              min_steps=args.min_steps,
                              object_bodies=scene.object_bodies)
        except Exception as e:
            skipped[f"extract failed: {type(e).__name__}"] += 1
            continue
        metas.extend(got)
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(runs)}] {len(metas)} episodes")

    manifest = {
        "n_episodes": len(metas),
        "outcomes_kept": args.outcomes,
        "scene": args.scene,
        "kp_arm": scene.kp_arm.tolist(),
        "episodes": [asdict(m) for m in metas],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n{len(metas)} episodes -> {args.out}")
    if skipped:
        print("skipped:", dict(skipped))

    by_method = Counter(m.method_key for m in metas)
    print("\nepisodes by collection method:")
    for k, v in by_method.most_common():
        steps = sum(m.n_steps for m in metas if m.method_key == k)
        print(f"  {k:42s} {v:4d} episodes  {steps:8d} steps")

    n_conflict = sum(m.method_conflict for m in metas)
    n_fallback = sum(m.action_source != "torque" for m in metas)
    print(f"\nmethod-label conflicts: {n_conflict}"
          f"   |   next_qpos action fallback: {n_fallback}")
    viol = [m.limit_viol_max_rad for m in metas if np.isfinite(m.limit_viol_max_rad)]
    if viol:
        print(f"recovered-target joint-limit violation: "
              f"p50={np.percentile(viol,50):.3f} max={max(viol):.3f} rad")
    rep = [m.grasp_repair_frac for m in metas]
    if any(rep):
        print(f"GRASP-phase finger-target repair: mean {np.mean(rep):.1%} of rows "
              f"({sum(r > 0 for r in rep)} episodes affected)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
