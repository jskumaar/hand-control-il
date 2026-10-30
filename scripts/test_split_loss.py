#!/usr/bin/env python3
"""Validation loss split into ARM (7 joints) and HAND (16 joints).

Why this exists: the headline comparison reports one pooled MSE over all 23
joints, so it cannot say WHERE the contact-aware data is harder to fit. Three
separate measurements already point at the arm rather than the fingers --

  * action divergence is 1.74x worse on the arm and 1.86x on the hand, but
    action VARIANCE is 1.80x worse on the arm and 0.80x (i.e. BETTER) on the
    hand -- the certificate appears to stabilise the squeeze while
    destabilising the approach;
  * training on the arm alone made the gap WORSE, 3.19x -> 8.82x;
  * the trajectories agree early and agree again at the grasp, and diverge in
    between -- which is arm motion, not finger motion.

If the per-block loss shows the gap concentrated in the arm, those three agree
and the hand-over account is supported. If the gap is in the hand, the
hand-over account is wrong and the cause is the GRASP-phase labels after all.
This is a cheap, direct test of that.

Method matches run_experiment.py exactly -- 8-fold CV over EPISODES, matched
episode counts, retargeter held fixed, the same pooled normaliser -- so the
numbers are comparable to the headline. The ONLY difference is that the
validation pass reports MSE over the arm slice and the hand slice separately
in addition to the pooled value.

Note the loss is MSE in NORMALISED action units (per-dimension z-score over
both conditions pooled). Arm and hand are separately normalised by their own
per-joint std, so the two blocks are directly comparable to each other even
though the underlying radian scales differ (arm std 0.28-0.89 rad, hand
0.12-0.71 rad).

Usage:

    python scripts/test_split_loss.py --data data/v3 --algo bc_mlp --seeds 6
    python scripts/test_split_loss.py --data data/v3 --algo bc_rnn --seeds 6
"""
from __future__ import annotations

import argparse
import json
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from handil.algos import build
from handil.data.dataset import episode_folds, manifest_obs_dim
from handil.data.schema import ACT_DIM, N_ARM, N_ROBOT
from handil.train import TrainConfig, make_datasets

CONDITIONS = ("baseline", "contact_aware")


def welch(a, b):
    """(a-b, Cohen d, two-sided p). Normal approximation for p."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan"), float("nan")
    eff = a.mean() - b.mean()
    sp = sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) or 1e-12
    se = sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) or 1e-12
    return eff, eff / sp, erfc(abs(eff / se) / sqrt(2))


def _pred(policy, obs):
    """One predicted action per sample, whatever the policy's output shape.

    BC-MLP returns (B, A); BC-RNN returns (B, A) from a window; ACT returns a
    chunk. We compare the FIRST predicted action against the first target, so
    every algorithm is scored on the same quantity.
    """
    if hasattr(policy, "predict_chunk"):
        o = obs if obs.ndim == 3 else obs.unsqueeze(1)
        out = policy.predict_chunk(o)
        return out[:, 0] if out.ndim == 3 else out
    out = policy(obs)
    return out[:, 0] if out.ndim == 3 else out


def fit_one_fold(ds, dim, algo, algo_kwargs, i_tr, i_va, seed, epochs, device,
                 batch_size=256, lr=1e-4, wd=1e-4):
    """Train one fold; return (pooled, arm, hand) MSE at the best pooled epoch.

    The three are taken from the SAME epoch -- the one that minimises the
    pooled loss -- so the split is a decomposition of the reported number, not
    three independently-optimised metrics.
    """
    torch.manual_seed(seed)
    policy = build(algo, obs_dim=dim, act_dim=ACT_DIM, **algo_kwargs).to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    dl = DataLoader(Subset(ds, i_tr), batch_size=batch_size, shuffle=True,
                    drop_last=True)
    dv = DataLoader(Subset(ds, i_va), batch_size=batch_size)

    best = (float("inf"), float("nan"), float("nan"))
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
        se_all = se_arm = se_hand = 0.0
        n = 0
        with torch.no_grad():
            for batch in dv:
                obs, act = [b.to(device) for b in batch]
                tgt = act[:, 0] if act.ndim == 3 else act
                pred = _pred(policy, obs)
                err2 = (pred - tgt) ** 2                      # (B, 23)
                se_all += err2.mean(1).sum().item()
                se_arm += err2[:, :N_ARM].mean(1).sum().item()
                se_hand += err2[:, N_ARM:N_ROBOT].mean(1).sum().item()
                n += tgt.shape[0]
        n = max(n, 1)
        pooled = se_all / n
        if pooled < best[0]:
            best = (pooled, se_arm / n, se_hand / n)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/v3"))
    ap.add_argument("--algo", default="bc_mlp")
    ap.add_argument("--retargeter", default="anyteleop")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = (("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else args.device)
    dim = manifest_obs_dim(args.data)
    manifest = json.loads((args.data / "manifest.json").read_text())

    counts = {}
    for pipe in CONDITIONS:
        counts[pipe] = sum(1 for e in manifest["episodes"]
                           if e["pipeline"] == pipe
                           and e["retargeter"] == args.retargeter
                           and not e.get("ablation")
                           and e["outcome"] == "success")
    n_eps = min(counts.values())

    print(f"data        {args.data}   algo {args.algo}   device {device}")
    print(f"retargeter  {args.retargeter} (fixed)")
    print(f"episodes    {counts} -> {n_eps} per condition")
    print(f"metric      {args.folds}-fold CV over EPISODES x {args.seeds} seeds")
    print(f"            MSE in normalised action units, split "
          f"arm[0:{N_ARM}] / hand[{N_ARM}:{N_ROBOT}]")
    print()

    res = {p: {"pooled": [], "arm": [], "hand": []} for p in CONDITIONS}
    for pipe in CONDITIONS:
        for seed in range(args.seeds):
            cfg = TrainConfig(
                algo=args.algo, data_dir=str(args.data),
                out_dir=str(Path("runs/_tmp_split")),
                pipeline=pipe, retargeter=args.retargeter,
                exclude_ablations=True, max_episodes=n_eps,
                epochs=args.epochs, seed=seed, algo_kwargs={})
            ds, _norm, _eps, _probe, ds_dim = make_datasets(cfg)
            vals = [fit_one_fold(ds, ds_dim, args.algo, {}, i_tr, i_va,
                                 seed, args.epochs, device)
                    for i_tr, i_va in episode_folds(ds, args.folds, seed)]
            pooled = float(np.mean([v[0] for v in vals]))
            arm = float(np.mean([v[1] for v in vals]))
            hand = float(np.mean([v[2] for v in vals]))
            res[pipe]["pooled"].append(pooled)
            res[pipe]["arm"].append(arm)
            res[pipe]["hand"].append(hand)
            print(f"  {pipe:14s} seed{seed:02d}  pooled {pooled:.5f}   "
                  f"arm {arm:.5f}   hand {hand:.5f}", flush=True)

    print("\n" + "=" * 74)
    print(f"ARM vs HAND -- {args.algo}, {args.folds}-fold CV, {args.seeds} seeds")
    print("-" * 74)
    print(f"  {'block':10s} {'baseline':>12s} {'contact_aware':>14s} "
          f"{'ratio':>8s} {'d':>8s} {'p':>10s}")
    summary = {}
    for block in ("pooled", "arm", "hand"):
        b = np.array(res["baseline"][block])
        c = np.array(res["contact_aware"][block])
        _eff, d, p = welch(b, c)
        ratio = c.mean() / b.mean() if b.mean() else float("nan")
        print(f"  {block:10s} {b.mean():12.5f} {c.mean():14.5f} "
              f"{ratio:7.2f}x {d:8.2f} {p:10.4f}")
        summary[block] = {"baseline": float(b.mean()),
                          "baseline_sd": float(b.std(ddof=1)) if len(b) > 1 else 0.0,
                          "contact_aware": float(c.mean()),
                          "contact_aware_sd": float(c.std(ddof=1)) if len(c) > 1 else 0.0,
                          "ratio": float(ratio), "cohen_d": float(d), "p": float(p)}

    arm_r, hand_r = summary["arm"]["ratio"], summary["hand"]["ratio"]
    print()
    if np.isfinite(arm_r) and np.isfinite(hand_r):
        if arm_r > hand_r:
            print(f"  -> The gap is concentrated in the ARM ({arm_r:.2f}x vs "
                  f"{hand_r:.2f}x).")
            print("     Consistent with the hand-over account: the approach is")
            print("     what varies, not the certified squeeze.")
        else:
            print(f"  -> The gap is concentrated in the HAND ({hand_r:.2f}x vs "
                  f"{arm_r:.2f}x).")
            print("     This CONTRADICTS the hand-over account and points back")
            print("     at the GRASP-phase labels.")

    out = args.out or Path(f"runs/split_loss_{args.algo}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"algo": args.algo, "data": str(args.data), "n_eps": n_eps,
         "seeds": args.seeds, "folds": args.folds,
         "per_seed": res, "summary": summary}, indent=2))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
