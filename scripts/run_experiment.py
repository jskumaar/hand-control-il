#!/usr/bin/env python3
"""The contact-aware vs baseline experiment, end to end.

One script, both conditions, with every methodological fix this project learned
the hard way already applied:

  * **K-fold CV over episodes**, not a single held-out split. A single split with
    val_frac=0.1 holds out 1-2 episodes, so the loss largely reports WHICH
    episode that was -- measured spread of 24x at 23 episodes, and seed sd of
    65-154% of the mean that does NOT shrink with more data. K-fold makes the
    metric 12x more stable (seed sd 71% -> 8% of the mean).
  * **Episode-level splits**, never per-sample. Windows ~10 timesteps apart in
    one episode are near-duplicates; splitting over samples put them on both
    sides and made losses look 25-85x better than they were.
  * **Matched episode counts** across conditions, so the comparison is not
    confounded by one arm simply having more data.
  * **Retargeter held fixed**, so pipeline is the only variable. Comparing
    contact_aware+vwj against baseline+anyteleop (as an earlier version of this
    work did) varies pipeline, retargeter and sample size at once.
  * **Closed-loop rollouts at >= 2500 steps.** At 400 steps every policy scores
    0%; the chunked policy's first real pick appears at 2500, and it plateaus
    there (4000 and 6000 are identical).

Usage:

    # the offline comparison, the headline number
    python scripts/run_experiment.py --data data/v2 --seeds 12

    # add closed-loop rollouts (slow: ~2500 steps x 4 objects x 2 conditions)
    python scripts/run_experiment.py --data data/v2 --seeds 12 --rollouts 4

    # a different algorithm, or the delta action space
    python scripts/run_experiment.py --algo act --data data/v2_delta --seeds 6
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from handil.algos import build
from handil.data.dataset import episode_folds, manifest_obs_dim
from handil.data.schema import ACT_DIM
from handil.train import TrainConfig, make_datasets

CONDITIONS = ("baseline", "contact_aware")


@dataclass
class Fit:
    pipeline: str
    seed: int
    cv_mean: float
    cv_sd: float
    folds: list[float]
    n_episodes: int


def fit_one_fold(ds, dim, algo, algo_kwargs, i_tr, i_va, seed, epochs,
                 device, batch_size=256, lr=1e-4, wd=1e-4) -> float:
    """Train on one fold, return the best validation loss seen."""
    torch.manual_seed(seed)
    policy = build(algo, obs_dim=dim, act_dim=ACT_DIM, **algo_kwargs).to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    dl = DataLoader(Subset(ds, i_tr), batch_size=batch_size, shuffle=True,
                    drop_last=True)
    dv = DataLoader(Subset(ds, i_va), batch_size=batch_size)
    best = float("inf")
    for _ in range(epochs):
        policy.train()
        for batch in dl:
            batch = [b.to(device) for b in batch]
            loss = policy.loss(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
        sched.step()
        policy.eval()
        tot = n = 0
        with torch.no_grad():
            for batch in dv:
                batch = [b.to(device) for b in batch]
                tot += policy.loss(batch).item() * len(batch[0])
                n += len(batch[0])
        best = min(best, tot / max(n, 1))
    return best


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """(effect, Cohen d, two-sided p) for a vs b. Normal approx for p."""
    eff = a.mean() - b.mean()
    sp = sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) or 1e-12
    se = sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) or 1e-12
    return eff, eff / sp, erfc(abs(eff / se) / sqrt(2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/v2"))
    ap.add_argument("--algo", default="bc_mlp")
    ap.add_argument("--retargeter", default="anyteleop",
                    help="held FIXED so pipeline is the only variable")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--folds", type=int, default=8, help="K in K-fold over episodes")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--episodes", type=int, default=None,
                    help="cap per condition; default = the smaller condition's count")
    ap.add_argument("--rollouts", type=int, default=0,
                    help="closed-loop rollouts per condition (0 = offline only)")
    ap.add_argument("--max-steps", type=int, default=2500,
                    help=">=2500: the first real pick appears there, and plateaus")
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("runs/experiment"))
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = (("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else args.device)
    if device == "cpu" and torch.cuda.device_count() > 0:
        print("[!] GPUs present but torch cannot use them -- this torch is built "
              f"for CUDA {torch.version.cuda}. Reinstall with `uv sync --extra cuda`.")
    algo_kwargs = {}
    if args.chunk is not None:
        algo_kwargs["chunk"] = args.chunk
    if args.horizon is not None:
        algo_kwargs["horizon"] = args.horizon

    manifest = json.loads((args.data / "manifest.json").read_text())
    dim = manifest_obs_dim(args.data)
    action_space = manifest.get("action_space", "position")

    # Match episode counts so the comparison is not confounded by data volume.
    counts = {}
    for pipe in CONDITIONS:
        counts[pipe] = sum(1 for e in manifest["episodes"]
                           if e["pipeline"] == pipe
                           and e["retargeter"] == args.retargeter
                           and not e.get("ablation")
                           and e["outcome"] == "success")
    n_eps = args.episodes or min(counts.values())
    if n_eps == 0:
        raise SystemExit(f"no episodes for retargeter={args.retargeter}: {counts}")

    print(f"data          {args.data}  (action_space={action_space}, obs_dim={dim})")
    print(f"algo          {args.algo} {algo_kwargs or ''}  device={device}")
    print(f"retargeter    {args.retargeter} (fixed)")
    print(f"episodes      available {counts} -> using {n_eps} per condition")
    print(f"metric        {args.folds}-fold CV over EPISODES x {args.seeds} seeds")
    print()

    args.out.mkdir(parents=True, exist_ok=True)
    fits: list[Fit] = []
    for pipe in CONDITIONS:
        for seed in range(args.seeds):
            cfg = TrainConfig(
                algo=args.algo, data_dir=str(args.data), out_dir=str(args.out / "_tmp"),
                pipeline=pipe, retargeter=args.retargeter, exclude_ablations=True,
                max_episodes=n_eps, epochs=args.epochs, seed=seed,
                algo_kwargs=algo_kwargs)
            ds, _norm, eps, _probe, ds_dim = make_datasets(cfg)
            vals = [fit_one_fold(ds, ds_dim, args.algo, algo_kwargs, i_tr, i_va,
                                 seed, args.epochs, device)
                    for i_tr, i_va in episode_folds(ds, args.folds, seed)]
            f = Fit(pipeline=pipe, seed=seed, cv_mean=float(np.mean(vals)),
                    cv_sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    folds=[float(v) for v in vals], n_episodes=len(eps))
            fits.append(f)
            print(f"  {pipe:14s} seed{seed:02d}  CV {f.cv_mean:.5f}  "
                  f"(fold sd {f.cv_sd:.5f})", flush=True)
            (args.out / "fits.json").write_text(
                json.dumps([vars(x) for x in fits], indent=2))

    b = np.array([f.cv_mean for f in fits if f.pipeline == "baseline"])
    c = np.array([f.cv_mean for f in fits if f.pipeline == "contact_aware"])
    eff, d, p = welch(b, c)

    print("\n" + "=" * 72)
    print(f"OFFLINE: {args.algo}, {args.folds}-fold CV over episodes, "
          f"{args.seeds} seeds, {n_eps} eps/condition")
    print("-" * 72)
    print(f"  baseline       {b.mean():.5f} +- {b.std(ddof=1):.5f}")
    print(f"  contact_aware  {c.mean():.5f} +- {c.std(ddof=1):.5f}")
    rel = 100 * eff / b.mean() if b.mean() else float("nan")
    print(f"  effect {eff:+.5f} ({rel:+.1f}% rel)  Cohen d {d:+.2f}  p ~ {p:.4f}")
    verdict = ("contact_aware better" if p < 0.05 and eff > 0 else
               "baseline better" if p < 0.05 and eff < 0 else
               "NOT significant")
    print(f"  verdict: {verdict}")
    if p >= 0.05 and abs(d) > 1e-6:
        need = 2 * ((1.96 + 0.84) / abs(d)) ** 2
        print(f"  (for 80% power at this effect size: ~{int(np.ceil(need))} seeds/arm)")

    summary = {"algo": args.algo, "data": str(args.data), "n_eps": n_eps,
               "seeds": args.seeds, "folds": args.folds,
               "baseline_mean": float(b.mean()), "baseline_sd": float(b.std(ddof=1)),
               "contact_aware_mean": float(c.mean()),
               "contact_aware_sd": float(c.std(ddof=1)),
               "effect": float(eff), "cohen_d": float(d), "p": float(p),
               "verdict": verdict}

    if args.rollouts:
        from handil.envs.rollout import PickPlaceEnv, rollout
        from handil.envs.scene import build_scene
        from handil.eval import WindowedPolicy
        from handil.train import load_policy, train

        print("\n" + "=" * 72)
        print(f"CLOSED-LOOP: {args.rollouts} rollouts/condition at "
              f"{args.max_steps} steps")
        print("-" * 72)
        scene = build_scene("robocasa")
        objs = list(scene.object_bodies)
        cl = {}
        for pipe in CONDITIONS:
            run_dir = args.out / f"cl_{pipe}"
            cfg = TrainConfig(
                algo=args.algo, data_dir=str(args.data), out_dir=str(run_dir),
                pipeline=pipe, retargeter=args.retargeter, exclude_ablations=True,
                max_episodes=n_eps, epochs=args.epochs, seed=0,
                algo_kwargs=algo_kwargs)
            train(cfg)
            policy = load_policy(run_dir, expect_dim=dim)
            kp = 0.8 if pipe == "contact_aware" else 2.0
            n_succ = n_pick = 0
            for i in range(args.rollouts):
                env = PickPlaceEnv(scene=scene, kp_hand=kp,
                                   active_object=objs[i % len(objs)],
                                   action_space=action_space)
                wp = WindowedPolicy(policy, getattr(policy, "horizon", 1))
                wp.reset()
                r = rollout(env, wp, max_steps=args.max_steps)
                n_succ += r.success
                n_pick += r.picked
            cl[pipe] = {"pick": n_pick / args.rollouts,
                        "success": n_succ / args.rollouts}
            print(f"  {pipe:14s} pick {cl[pipe]['pick']:.0%}  "
                  f"success {cl[pipe]['success']:.0%}")
        summary["closed_loop"] = cl

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n-> {args.out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
