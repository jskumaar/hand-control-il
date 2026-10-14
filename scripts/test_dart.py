#!/usr/bin/env python3
"""Does DART-style noise injection give non-zero closed-loop success?

The decision this answers: plain BC drifts 0.088 rad off the demonstrated path
after one step and has no recovery data (nearest other-episode state is 0.31 rad
away). DART synthesises that recovery data from existing demos. If it produces
non-zero success, collecting demos from jittered starts will work and this tells
us roughly how much spread is needed. If it does not, the gap is bigger than
coverage and more demos may not rescue it either.

    python scripts/test_dart.py --data data/v2 --episodes 12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from handil.data.dataset import manifest_obs_dim
from handil.envs.rollout import PickPlaceEnv
from handil.envs.scene import build_scene
from handil.envs.rollout import rollout
from handil.eval import WindowedPolicy
from handil.train import TrainConfig, load_policy, train


def drift_and_reach(policy, scene, obj, kp_hand, ref_obs, steps=300):
    """Closed-loop drift from a reference demo, and closest hand-object approach."""
    from handil.data.schema import obs_slice
    env = PickPlaceEnv(scene=scene, kp_hand=kp_hand, active_object=obj)
    wp = WindowedPolicy(policy, getattr(policy, "horizon", 1))
    wp.reset()
    obs = env.reset()
    sl = obs_slice("obj_in_palm")
    dev, oip = [], []
    for i in range(steps):
        obs = env.step(wp(obs))
        oip.append(float(np.linalg.norm(obs[sl])))
        if i < len(ref_obs):
            dev.append(float(np.linalg.norm(obs[:7] - ref_obs[i, :7])))
    return dev, min(oip)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/v2"))
    ap.add_argument("--pipeline", default="contact_aware")
    ap.add_argument("--retargeter", default="anyteleop")
    ap.add_argument("--algos", nargs="+", default=["bc_rnn"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--episodes", type=int, default=12, help="eval rollouts/cell")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--out", type=Path, default=Path("runs/dart"))
    ap.add_argument("--sweep", nargs="*", default=None,
                    help="copies:sigma pairs, e.g. 0:0 2:0.05 4:0.10")
    args = ap.parse_args()

    sweep = [(0, 0.0)]
    for s in (args.sweep or ["2:0.02", "2:0.05", "4:0.05", "4:0.10", "8:0.10"]):
        c, g = s.split(":")
        sweep.append((int(c), float(g)))

    scene = build_scene("robocasa")
    kp_hand = 0.8 if args.pipeline == "contact_aware" else 2.0
    dim = manifest_obs_dim(args.data)

    eps = json.loads((args.data / "manifest.json").read_text())["episodes"]
    ref = [e for e in eps if e["pipeline"] == args.pipeline
           and e["retargeter"] == args.retargeter]
    ref_obs = np.load(args.data / f"{ref[0]['episode_id']}.npz",
                      allow_pickle=True)["obs"]
    objs = list(scene.object_bodies)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for algo in args.algos:
        for copies, sigma in sweep:
            tag = f"{algo}_c{copies}_s{sigma:g}"
            cfg = TrainConfig(
                algo=algo, data_dir=str(args.data), out_dir=str(args.out / tag),
                pipeline=args.pipeline, retargeter=args.retargeter,
                exclude_ablations=True, epochs=args.epochs, seed=0,
                dart_copies=copies, dart_sigma_arm=sigma)
            print(f"\n=== {tag} ===", flush=True)
            s = train(cfg)
            policy = load_policy(args.out / tag, expect_dim=dim)

            n_succ = n_pick = 0
            for i in range(args.episodes):
                env = PickPlaceEnv(scene=scene, kp_hand=kp_hand,
                                   active_object=objs[i % len(objs)])
                wp = WindowedPolicy(policy, getattr(policy, "horizon", 1))
                wp.reset()
                r = rollout(env, wp, max_steps=args.max_steps)
                n_succ += r.success
                n_pick += r.picked
            dev, reach = drift_and_reach(policy, scene, objs[0], kp_hand, ref_obs)
            row = dict(algo=algo, copies=copies, sigma=sigma,
                       val=s["best_val"], n_samples=s["n_samples"],
                       success=n_succ / args.episodes, pick=n_pick / args.episodes,
                       drift50=dev[50] if len(dev) > 50 else None,
                       drift200=dev[min(200, len(dev) - 1)] if dev else None,
                       min_oip=reach)
            rows.append(row)
            print(f"  val {row['val']:.5f}  success {row['success']:.0%}  "
                  f"pick {row['pick']:.0%}  drift@50 {row['drift50']:.2f}  "
                  f"drift@200 {row['drift200']:.2f}  min|oip| {reach:.3f}",
                  flush=True)
            (args.out / "results.json").write_text(json.dumps(rows, indent=2))

    print("\n" + "=" * 86)
    print(f"{'algo':8s} {'copies':>6s} {'sigma':>6s} {'val':>9s} {'samples':>8s} "
          f"{'pick':>6s} {'succ':>6s} {'drift@50':>9s} {'drift@200':>10s} {'reach':>7s}")
    print("-" * 86)
    for r in rows:
        print(f"{r['algo']:8s} {r['copies']:6d} {r['sigma']:6.2f} {r['val']:9.5f} "
              f"{r['n_samples']:8d} {r['pick']:5.0%} {r['success']:5.0%} "
              f"{r['drift50']:9.2f} {r['drift200']:10.2f} {r['min_oip']:7.3f}")
    print(f"\n-> {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
