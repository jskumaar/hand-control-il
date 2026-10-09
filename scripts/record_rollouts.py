#!/usr/bin/env python3
"""Record MP4 video of a trained policy, or of a demonstration replay.

Policies currently fail the task, so the videos are a diagnostic: they show
*how* a rollout fails (drifts past the object, closes on air, knocks it over)
in a way the drift and reach numbers cannot.

    # a trained policy, one episode per object
    python scripts/record_rollouts.py --run runs/pos_ca_bcrnn --out videos/

    # the demonstrations themselves, for side-by-side comparison
    python scripts/record_rollouts.py --demos --data data/v2 --limit 4 --out videos/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from handil.envs.rollout import PickPlaceEnv
from handil.envs.scene import build_scene
from handil.eval import RolloutRecorder, WindowedPolicy, replay_episode
from handil.eval.record import DEFAULT_CAMERAS


def _record_demo(env, ep, data_dir, rec):
    """Replay one demonstration's recorded actions, capturing frames."""
    d = np.load(data_dir / f"{ep['episode_id']}.npz", allow_pickle=True)
    # replay_episode owns the stepping loop, so hook capture onto env.step.
    orig = env.step

    def step_and_capture(a):
        o = orig(a)
        rec.capture(env.data)
        return o

    env.step = step_and_capture
    try:
        return replay_episode(env, d["action"], obs_ref=d["obs"], t=d["t"],
                              episode_id=ep["episode_id"], obj=ep["obj"])
    finally:
        env.step = orig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", type=Path, help="a training run dir to roll out")
    src.add_argument("--demos", action="store_true",
                     help="replay recorded demonstrations instead of a policy")
    ap.add_argument("--data", type=Path, default=Path("data/v2"))
    ap.add_argument("--out", type=Path, default=Path("videos"))
    ap.add_argument("--objects", nargs="*", default=None,
                    help="object body names; default: one episode per object")
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--kp-hand", type=float, default=None)
    ap.add_argument("--squeeze", action="store_true")
    ap.add_argument("--gamma", type=float, default=5.0)
    ap.add_argument("--cameras", nargs="*", default=list(DEFAULT_CAMERAS))
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--size", default="1280x960")
    ap.add_argument("--root", type=Path, default=None)
    args = ap.parse_args()

    w, h = (int(v) for v in args.size.lower().split("x"))
    scene = build_scene("robocasa", root=args.root)
    args.out.mkdir(parents=True, exist_ok=True)

    eps = json.loads((args.data / "manifest.json").read_text())["episodes"]

    if args.demos:
        # One episode per distinct object, so the videos cover the task range.
        picked, seen = [], set()
        for e in eps:
            if e["obj"] in seen:
                continue
            seen.add(e["obj"])
            picked.append(e)
            if len(picked) >= args.limit:
                break
        jobs = [(e["obj"], e) for e in picked]
        tag = "demo"
        policy = None
    else:
        from handil.train import load_policy
        policy = load_policy(args.run, device="cpu")
        cfg = json.loads((args.run / "config.json").read_text())
        if args.kp_hand is None:
            args.kp_hand = 0.8 if cfg.get("pipeline") == "contact_aware" else 2.0
        names = args.objects or list(scene.object_bodies)[: args.limit]
        jobs = [(n, None) for n in names]
        tag = args.run.name

    if args.kp_hand is None:
        args.kp_hand = 2.0

    print(f"recording {len(jobs)} rollout(s) of {tag} -> {args.out}")
    results = []
    for obj, ep in jobs:
        env = PickPlaceEnv(scene=scene, kp_hand=args.kp_hand, active_object=obj,
                           squeeze=args.squeeze, gamma=args.gamma)
        name = f"{tag}__{obj}"
        rec = RolloutRecorder(env.model, args.out, name, cameras=args.cameras,
                              fps=args.fps, size=(w, h))
        if not rec.enabled:
            print("  recording unavailable; aborting")
            return 1

        if ep is not None:
            r = _record_demo(env, ep, args.data, rec)
            outcome, maxh = ("success" if r.success else
                             "pick" if r.picked else "-"), r.max_height
        else:
            wp = WindowedPolicy(policy, getattr(policy, "horizon", 1))
            wp.reset()
            obs = env.reset()
            rec.capture(env.data, force=True)
            maxh = -np.inf
            for _ in range(args.max_steps):
                obs = env.step(wp(obs))
                rec.capture(env.data)
                maxh = max(maxh, env.object_height_above_rest())
            outcome = "timeout"

        info = rec.close()
        print(f"  {obj:<22} {outcome:<8} max_h={maxh:+.3f}  "
              f"{info.n_frames} frames")
        for cam, p in info.paths.items():
            print(f"      {cam:<9} {p}  ({p.stat().st_size/1e6:.1f} MB)")
        results.append({"object": obj, "outcome": outcome,
                        "max_height": float(maxh),
                        "videos": {k: str(v) for k, v in info.paths.items()}})

    (args.out / f"{tag}_index.json").write_text(json.dumps(results, indent=2))
    print(f"\n-> {args.out / f'{tag}_index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
