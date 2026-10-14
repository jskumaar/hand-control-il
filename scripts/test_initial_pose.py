#!/usr/bin/env python3
"""Does a policy generalise off the single demonstrated start pose?

Every demonstration starts from a bit-identical home pose -- zero spread across
all 7 arm joints -- which is the measured root cause of BC's closed-loop failure
here. This measures the consequence directly, by evaluating a trained policy
from jittered starts and watching how fast competence decays.

Two things it answers:

  * How much start-pose variation a policy tolerates. If drift is already large
    at 0 rad of jitter (it is: 1.6 rad by step 50), the curve tells us whether
    jitter makes it meaningfully worse or whether the policy was never tracking
    the demonstration in the first place.
  * Whether DART helps where it should. A noise-injected policy has seen
    off-path states, so it should degrade MORE SLOWLY with jitter than a plain
    BC policy. That is the cleanest test of whether the synthetic recovery data
    actually transferred.

    python scripts/test_initial_pose.py --runs runs/dart/bc_rnn_c0_s0 runs/dart/bc_rnn_c4_s0.05
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from handil.data.dataset import manifest_obs_dim
from handil.data.schema import obs_slice
from handil.envs.rollout import (ARRIVAL_DWELL_S, ARRIVAL_SPEED_M_S, DWELL_S,
                                 LIFT_HEIGHT_M, PickPlaceEnv)
from handil.envs.scene import build_scene
from handil.eval import WindowedPolicy
from handil.train import load_policy


def score(env, policy, max_steps, jitter, obj, rng):
    """One rollout from a jittered start; returns (success, picked, max_h, reach)."""
    wp = WindowedPolicy(policy, getattr(policy, "horizon", 1))
    wp.reset()
    env.set_active_object(obj)
    obs = env.reset(arm_jitter=jitter, rng=rng)
    sl = obs_slice("obj_in_palm")
    dwell = arrive = None
    picked = success = False
    max_h = -np.inf
    reach = np.inf
    for _ in range(max_steps):
        obs = env.step(wp(obs))
        t = float(env.data.time)
        reach = min(reach, float(np.linalg.norm(obs[sl])))
        h = env.object_height_above_rest()
        max_h = max(max_h, h)
        held = env.hand_touching(env.active_object)
        in_bowl = env.object_in_bowl(env.active_object)
        if not picked:
            if h > LIFT_HEIGHT_M and held:
                dwell = t if dwell is None else dwell
                if t - dwell >= DWELL_S:
                    picked = True
            else:
                dwell = None
        if in_bowl and env.object_speed() < ARRIVAL_SPEED_M_S and not held:
            arrive = t if arrive is None else arrive
            if t - arrive >= ARRIVAL_DWELL_S:
                success = True
                break
        else:
            arrive = None
    return success, picked, float(max_h), float(reach)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="training run dirs")
    ap.add_argument("--data", type=Path, default=Path("data/v2"))
    ap.add_argument("--jitters", nargs="*", type=float,
                    default=[0.0, 0.02, 0.05, 0.10, 0.20])
    ap.add_argument("--episodes", type=int, default=8, help="rollouts per jitter")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--kp-hand", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs/initpose.json"))
    args = ap.parse_args()

    scene = build_scene("robocasa")
    dim = manifest_obs_dim(args.data)
    objs = list(scene.object_bodies)
    rows = []

    for run in args.runs:
        run = Path(run)
        policy = load_policy(run, expect_dim=dim)
        print(f"\n=== {run.name} ===", flush=True)
        for j in args.jitters:
            rng = np.random.default_rng(args.seed)
            env = PickPlaceEnv(scene=scene, kp_hand=args.kp_hand,
                               active_object=objs[0])
            res = [score(env, policy, args.max_steps, j, objs[i % len(objs)], rng)
                   for i in range(args.episodes)]
            succ = np.mean([r[0] for r in res])
            pick = np.mean([r[1] for r in res])
            mh = np.mean([r[2] for r in res])
            reach = np.mean([r[3] for r in res])
            rows.append(dict(run=run.name, jitter=j, success=float(succ),
                             pick=float(pick), mean_max_h=mh, mean_reach=reach))
            print(f"  jitter {j:.2f} rad  pick {pick:.0%}  success {succ:.0%}  "
                  f"max_h {mh:+.3f}  reach {reach:.3f}", flush=True)
            args.out.write_text(json.dumps(rows, indent=2))

    print("\n" + "=" * 72)
    print(f"{'run':30s} {'jitter':>7s} {'pick':>6s} {'succ':>6s} {'reach':>7s}")
    print("-" * 72)
    for r in rows:
        print(f"{r['run']:30s} {r['jitter']:7.2f} {r['pick']:5.0%} "
              f"{r['success']:5.0%} {r['mean_reach']:7.3f}")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
