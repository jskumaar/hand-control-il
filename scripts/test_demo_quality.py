#!/usr/bin/env python3
"""Demonstration-quality metrics from the IL literature, baseline vs contact-aware.

Motivation: the chain this project tests is

    certified grasps -> better demonstrations -> better policies

Link 2->3 is well supported in the literature (robomimic CoRL 2021: 300
multi-human demos train a WORSE policy than 200 proficient-human ones;
Belkhale et al. NeurIPS 2023 formalise why). But this project only ever measured
the ENDS of the chain -- grasp certification on one side, policy loss on the
other -- so a failure of the chain could not be localised. This script measures
the MIDDLE: is the contact-aware data better *as data*, by the field's own
metrics, independent of any policy?

That matters because the two answers have opposite consequences:

  * demos measure better but policies train worse  -> the metrics (and the
    premise behind them) do not transfer to this setting; the learnability
    inversion is the interesting finding.
  * demos measure worse                            -> link 1->2 simply fails,
    and the policy result is downstream of a data problem we can name.

Metrics, each with its provenance, and each a QUANTITY WHERE LOWER IS BETTER
unless noted:

  action_divergence   Belkhale, Cui & Sadigh, "Data Quality in Imitation
                      Learning", NeurIPS 2023 (arXiv:2306.02437). Their central
                      quantity: disagreement between demonstrated actions at
                      *nearby states*. Estimated here as the mean pairwise
                      action distance among k nearest neighbours in normalised
                      observation space, restricted to CROSS-EPISODE pairs --
                      within-episode neighbours are consecutive frames and
                      would report the timestep, not the disagreement.
                      Their Thm 4.1 carries an (H-t) time-to-go weight, so we
                      also report it weighted.

  action_variance     The same paper's practical diagnostic (per-cluster action
                      variance). Reported for arm and hand separately: this
                      project's earlier measurement found the hand plateaus at
                      0.035-0.058 rad even at near-identical observations while
                      the arm extrapolates to ~0.002-0.008, so pooling them
                      hides the effect.

  horizon             Belkhale et al. and robomimic both find longer demos
                      correlate with worse policies. Here: episode length.

  state_diversity     Belkhale et al. report this is NOT correlated with policy
                      success -- included precisely so we do not reward a
                      condition for mere coverage. HIGHER is more diverse; no
                      implied direction.

  spectral_arc_length Smoothness. RINSE (arXiv:2604.23000) proposes SAL for
                      demo curation; arXiv:2610.17427 then shows SAL may track
                      episode LENGTH rather than skill (AUROC 0.952 vs 0.957
                      for length alone; 0.628 once duration-matched). So SAL is
                      reported ONLY alongside a duration-matched control, and
                      must not be read on its own. Both of those are short
                      single-author preprints, not peer-reviewed.

  mean_jerk           The baseline RINSE argues against, which the audit found
                      *outperforms* SAL under duration matching (0.896). Cheap,
                      so reported.

Deliberately NOT computed: any metric that would need a trained policy (that is
the thing we are trying to predict), and mutual-information estimates a la
DemInf (arXiv:2502.08623) -- k-NN MI over a VAE embedding needs more episodes
per cell than the 31-41 available here to be anything but noise.

Usage:

    python scripts/test_demo_quality.py --data data/v3
    python scripts/test_demo_quality.py --data data/v3 --retargeter vwj
"""
from __future__ import annotations

import argparse
import json
from math import erfc, sqrt
from pathlib import Path

import numpy as np

from handil.data.dataset import EpisodeStore, Normalizer, load_manifest, select
from handil.data.schema import N_ARM, N_ROBOT, obs_slice

CONDITIONS = ("baseline", "contact_aware")


# --------------------------------------------------------------------------
# statistics


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """(a-b, Cohen d, two-sided p). Normal approximation for p."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan"), float("nan")
    eff = a.mean() - b.mean()
    sp = sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) or 1e-12
    se = sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) or 1e-12
    return eff, eff / sp, erfc(abs(eff / se) / sqrt(2))


# --------------------------------------------------------------------------
# metrics


def action_divergence(store: EpisodeStore, norm: Normalizer, k: int = 8,
                      max_q: int = 4000, seed: int = 0) -> dict:
    """Mean action disagreement among cross-episode observation neighbours.

    Belkhale et al.'s action divergence is defined against the expert's own
    distribution; with a single demonstration set the computable analogue is
    disagreement between demonstrators (here: between episodes) at states the
    policy would treat as equivalent.

    Cross-episode only. Within an episode the nearest neighbours of a frame are
    its temporal neighbours, whose actions differ by one control step (~0.006
    rad for the arm here) -- that reports dt, not inconsistency.

    Returns arm/hand means plus the (H-t)-weighted variant from their Thm 4.1,
    where early-timestep divergence is weighted by time-to-go.
    """
    rng = np.random.default_rng(seed)
    obs = [np.nan_to_num(norm.norm_obs(o)) for o in store.obs]
    act = store.act
    ep_of = np.concatenate([np.full(len(o), i) for i, o in enumerate(obs)])
    t_frac = np.concatenate([np.linspace(0, 1, len(o)) for o in obs])
    O = np.concatenate(obs)
    A = np.concatenate(act)

    n = len(O)
    q_idx = rng.choice(n, size=min(max_q, n), replace=False)

    arm_d, hand_d, w_d, weights = [], [], [], []
    for qi in q_idx:
        other = ep_of != ep_of[qi]
        if not other.any():
            continue
        d = np.linalg.norm(O[other] - O[qi], axis=1)
        pool = np.flatnonzero(other)
        nn = pool[np.argsort(d)[:k]]
        da = A[nn] - A[qi]
        arm = float(np.abs(da[:, :N_ARM]).mean())
        hand = float(np.abs(da[:, N_ARM:N_ROBOT]).mean())
        arm_d.append(arm)
        hand_d.append(hand)
        # Thm 4.1's (H-t) weight, normalised: divergence early in the episode
        # propagates for longer.
        w = 1.0 - t_frac[qi]
        w_d.append(w * (arm + hand))
        weights.append(w)

    tot_w = sum(weights) or 1.0
    return {"arm": float(np.mean(arm_d)), "hand": float(np.mean(hand_d)),
            "weighted": float(sum(w_d) / tot_w), "n_queries": len(arm_d)}


def action_variance(store: EpisodeStore, norm: Normalizer, k: int = 8,
                    max_q: int = 4000, seed: int = 0) -> dict:
    """Per-neighbourhood action variance (Belkhale et al.'s practical metric).

    Same neighbourhoods as action_divergence, but variance of the neighbours'
    actions rather than distance from the query's action -- the form their
    robomimic analysis uses.
    """
    rng = np.random.default_rng(seed)
    obs = [np.nan_to_num(norm.norm_obs(o)) for o in store.obs]
    ep_of = np.concatenate([np.full(len(o), i) for i, o in enumerate(obs)])
    O = np.concatenate(obs)
    A = np.concatenate(store.act)

    n = len(O)
    q_idx = rng.choice(n, size=min(max_q, n), replace=False)
    arm_v, hand_v = [], []
    for qi in q_idx:
        other = ep_of != ep_of[qi]
        if not other.any():
            continue
        d = np.linalg.norm(O[other] - O[qi], axis=1)
        pool = np.flatnonzero(other)
        nn = pool[np.argsort(d)[:k]]
        arm_v.append(float(A[nn][:, :N_ARM].var(0).mean()))
        hand_v.append(float(A[nn][:, N_ARM:N_ROBOT].var(0).mean()))
    return {"arm": float(np.mean(arm_v)), "hand": float(np.mean(hand_v))}


def spectral_arc_length(traj: np.ndarray, dt: float = 0.02) -> float:
    """SAL smoothness of a multi-DOF trajectory (more negative = less smooth).

    Balasubramanian et al.'s measure, as used by RINSE for demo curation. Read
    this ONLY against the duration-matched control: arXiv:2610.17427 reports it
    tracks episode length rather than skill.
    """
    v = np.diff(traj, axis=0) / dt
    speed = np.linalg.norm(v, axis=1)
    if len(speed) < 8 or not np.isfinite(speed).all() or speed.max() <= 0:
        return float("nan")
    nfft = int(2 ** np.ceil(np.log2(len(speed)) + 2))
    mag = np.abs(np.fft.rfft(speed, n=nfft))
    mag = mag / (mag.max() or 1.0)
    f = np.fft.rfftfreq(nfft, d=dt)
    cut = f <= 10.0           # 10 Hz, the conventional movement band
    f, mag = f[cut], mag[cut]
    if len(f) < 2:
        return float("nan")
    df = np.diff(f) / (f[-1] or 1.0)
    dm = np.diff(mag)
    return float(-np.sum(np.sqrt(df ** 2 + dm ** 2)))


def mean_jerk(traj: np.ndarray, dt: float = 0.02) -> float:
    """Mean |third derivative| -- the baseline RINSE argues against, and which
    the audit paper found beats SAL once duration is controlled."""
    if len(traj) < 4:
        return float("nan")
    j = np.diff(traj, n=3, axis=0) / dt ** 3
    return float(np.abs(j).mean())


def per_episode(store: EpisodeStore) -> dict:
    """Episode-level metrics: horizon, smoothness, jerk, state diversity."""
    arm = obs_slice("q_arm")
    out = {"horizon": [], "sal": [], "jerk": [], "path_len": []}
    for o, e in zip(store.obs, store.meta):
        q = np.nan_to_num(o[:, arm])
        dt = float(e.get("dt_median") or 0.02)
        out["horizon"].append(len(q))
        out["sal"].append(spectral_arc_length(q, dt))
        out["jerk"].append(mean_jerk(q, dt))
        out["path_len"].append(float(np.abs(np.diff(q, axis=0)).sum()))
    return out


def state_diversity(store: EpisodeStore, norm: Normalizer) -> float:
    """Mean pairwise distance between episode means in normalised obs space.

    Included as a control, NOT as a quality target: Belkhale et al. report state
    diversity is uncorrelated with policy success, and warn that coverage bought
    from a noisier demonstrator can cost more in action divergence than it buys.
    """
    cent = np.stack([np.nan_to_num(norm.norm_obs(o)).mean(0) for o in store.obs])
    if len(cent) < 2:
        return float("nan")
    d = [np.linalg.norm(cent[i] - cent[j])
         for i in range(len(cent)) for j in range(i + 1, len(cent))]
    return float(np.mean(d))


# --------------------------------------------------------------------------


def duration_matched(a_vals, a_len, b_vals, b_len, tol=0.15):
    """Subsample both conditions to a common episode-length band.

    The audit (arXiv:2610.17427) found SAL's apparent discrimination was largely
    episode length: AUROC 0.952 unmatched vs 0.957 for length alone, falling to
    0.628 inside a duration-matched band. So any smoothness claim here has to
    survive this.
    """
    a_len, b_len = np.asarray(a_len, float), np.asarray(b_len, float)
    lo = max(np.percentile(a_len, 10), np.percentile(b_len, 10))
    hi = min(np.percentile(a_len, 90), np.percentile(b_len, 90))
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        return None
    ma, mb = (a_len >= lo) & (a_len <= hi), (b_len >= lo) & (b_len <= hi)
    if ma.sum() < 3 or mb.sum() < 3:
        return None
    return (np.asarray(a_vals, float)[ma], np.asarray(b_vals, float)[mb],
            (lo, hi), int(ma.sum()), int(mb.sum()))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/v3"))
    ap.add_argument("--retargeter", default="anyteleop",
                    help="held FIXED so pipeline is the only variable")
    ap.add_argument("--k", type=int, default=8, help="neighbours per query")
    ap.add_argument("--queries", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs/demo_quality.json"))
    args = ap.parse_args()

    episodes = load_manifest(args.data)
    res: dict[str, dict] = {}
    stores: dict[str, EpisodeStore] = {}

    # One normaliser over BOTH conditions: per-condition normalisation would
    # rescale each arm by its own spread and make the comparison meaningless.
    both = [e for pipe in CONDITIONS
            for e in select(episodes, pipeline=pipe, retargeter=args.retargeter)]
    norm = EpisodeStore(args.data, both).fit_normalizer()

    counts = {}
    for pipe in CONDITIONS:
        sel = select(episodes, pipeline=pipe, retargeter=args.retargeter)
        counts[pipe] = len(sel)
    n_eps = min(counts.values())
    if n_eps < 3:
        raise SystemExit(f"too few episodes for retargeter={args.retargeter}: {counts}")

    print(f"data        {args.data}")
    print(f"retargeter  {args.retargeter} (fixed)")
    print(f"episodes    available {counts} -> using {n_eps} per condition")
    print(f"neighbours  k={args.k}, {args.queries} queries, cross-episode only")
    print()

    rng = np.random.default_rng(args.seed)
    for pipe in CONDITIONS:
        sel = select(episodes, pipeline=pipe, retargeter=args.retargeter)
        # Match counts, as run_experiment.py does, so volume is not a factor.
        if len(sel) > n_eps:
            keep = rng.choice(len(sel), size=n_eps, replace=False)
            sel = [sel[i] for i in sorted(keep)]
        store = EpisodeStore(args.data, sel)
        stores[pipe] = store
        r = {"n_episodes": store.n_episodes,
             "divergence": action_divergence(store, norm, args.k, args.queries,
                                             args.seed),
             "variance": action_variance(store, norm, args.k, args.queries,
                                         args.seed),
             "state_diversity": state_diversity(store, norm)}
        r.update(per_episode(store))
        res[pipe] = r
        dv, vv = r["divergence"], r["variance"]
        print(f"  {pipe:14s} n={store.n_episodes}  "
              f"div(arm) {dv['arm']:.5f}  div(hand) {dv['hand']:.5f}  "
              f"var(hand) {vv['hand']:.5f}")

    b, c = res["baseline"], res["contact_aware"]
    print("\n" + "=" * 74)
    print("DEMONSTRATION QUALITY -- baseline vs contact_aware")
    print("lower is better except state_diversity (no implied direction)")
    print("-" * 74)

    rows = [
        ("action_divergence arm", b["divergence"]["arm"], c["divergence"]["arm"], True),
        ("action_divergence hand", b["divergence"]["hand"], c["divergence"]["hand"], True),
        ("  ... (H-t) weighted", b["divergence"]["weighted"],
         c["divergence"]["weighted"], True),
        ("action_variance arm", b["variance"]["arm"], c["variance"]["arm"], True),
        ("action_variance hand", b["variance"]["hand"], c["variance"]["hand"], True),
        ("state_diversity", b["state_diversity"], c["state_diversity"], False),
    ]
    print(f"  {'metric':26s} {'baseline':>11s} {'contact_aw':>11s} {'ratio':>8s}  better")
    for name, bv, cv, directional in rows:
        ratio = cv / bv if bv else float("nan")
        if not directional:
            who = "--"
        else:
            who = "contact_aware" if cv < bv else "baseline"
        print(f"  {name:26s} {bv:11.5f} {cv:11.5f} {ratio:8.2f}x  {who}")

    # Episode-level metrics get a test across episodes.
    print()
    summary_tests = {}
    for key, label in (("horizon", "horizon (steps)"),
                       ("jerk", "mean jerk"),
                       ("path_len", "arm path length (rad)")):
        eff, d, p = welch(np.array(b[key]), np.array(c[key]))
        bm, cm = np.nanmean(b[key]), np.nanmean(c[key])
        who = "contact_aware" if cm < bm else "baseline"
        print(f"  {label:26s} {bm:11.4f} {cm:11.4f} "
              f"{'':8s}  {who}  d={d:+.2f} p={p:.4f}")
        summary_tests[key] = {"baseline": float(bm), "contact_aware": float(cm),
                              "cohen_d": float(d), "p": float(p)}

    # SAL, with the duration-matched control it must not be read without.
    print("\n  spectral arc length (NOT to be read without the control below):")
    bm, cm = np.nanmean(b["sal"]), np.nanmean(c["sal"])
    eff, d, p = welch(np.array(b["sal"]), np.array(c["sal"]))
    print(f"    unmatched              {bm:11.4f} {cm:11.4f} "
          f"         d={d:+.2f} p={p:.4f}")
    dm = duration_matched(b["sal"], b["horizon"], c["sal"], c["horizon"])
    if dm is None:
        print("    duration-matched       insufficient overlap -- SAL UNUSABLE here")
        summary_tests["sal"] = {"matched": None}
    else:
        av, bv, band, na, nb = dm
        eff2, d2, p2 = welch(av, bv)
        print(f"    duration-matched       {np.nanmean(av):11.4f} "
              f"{np.nanmean(bv):11.4f}          d={d2:+.2f} p={p2:.4f}")
        print(f"      (band {band[0]:.0f}-{band[1]:.0f} steps, "
              f"n={na} vs {nb}; d fell {abs(d):.2f} -> {abs(d2):.2f})")
        summary_tests["sal"] = {
            "unmatched_d": float(d), "unmatched_p": float(p),
            "matched_d": float(d2), "matched_p": float(p2),
            "band": [float(band[0]), float(band[1])], "n": [na, nb]}

    print("\n" + "-" * 74)
    print("How to read this:")
    print("  * Belkhale et al. (NeurIPS 2023) predict LOWER action divergence")
    print("    and variance -> better policies, and that state diversity does")
    print("    NOT predict performance. So the divergence rows are the ones")
    print("    that speak to the 'better demos' link.")
    print("  * SAL and jerk come from the curation-metric line, which is")
    print("    CONTESTED as of 2026-10 and so far only in non-peer-reviewed")
    print("    preprints (arXiv:2610.17427, arXiv:2606.10229). Hence the")
    print("    duration-matched control.")
    print("  * This script says nothing about policies. Compare it against")
    print("    run_experiment.py: if demos measure better here but train worse")
    print("    there, the premise -- not the pipeline -- is what failed.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"data": str(args.data), "retargeter": args.retargeter,
         "k": args.k, "n_eps_per_condition": n_eps,
         "conditions": res, "tests": summary_tests}, indent=2))
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
