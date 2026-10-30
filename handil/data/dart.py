"""DART-style noise injection: synthesise recovery data from existing demos.

The measured failure of plain BC here is coverage, not label noise:

  * The policy predicts the demonstrated action to 0.007 rad when fed the
    demonstration's own observations -- it fits.
  * Rolled out it is 0.088 rad off the demonstrated path after ONE step, 1.12 rad
    by step 50.
  * Along-trajectory sample spacing is 0.006 rad while the nearest OTHER-episode
    state is 0.31 rad away -- 50x sparser between trajectories than along them.
  * So the first deviation puts the policy off-manifold, and no demonstration
    says how to come back, because no demonstrator was ever off.

DART (Laskey et al. 2017) attacks exactly this: perturb the states the
demonstrator visited and label each perturbed state with the action that returns
to the demonstrated trajectory. The policy then sees (off-path state -> recovery
action) pairs it would otherwise never get.

Why relabelling is sound HERE, and only for the arm. The matched-state analysis
showed the arm's cross-episode action disagreement extrapolates to 0.002-0.008
rad as observations coincide, with correlation 0.85-0.90 against observation
distance -- the demonstrators agree about the arm, they just do not co-visit
states. The hand does NOT have that property: its disagreement plateaus at
0.035-0.073 rad even at identical observations. So a synthetic arm label is
well-defined and a synthetic hand label is not; the hand action is copied
unchanged rather than invented.

The recovery label is exact rather than learned, because the controller is
known: the demonstrated target at time t is already in the dataset, and the
P-controller reaches it from ANY nearby state. Perturbing the state while
keeping the demonstrated target IS the recovery action -- the resulting torque
Kp*(target - perturbed_qpos) is larger precisely in the direction that undoes
the perturbation.

Wrap-around is not an issue on this data: joints 2 and 6 sit near pi for 58-73%
of rows but stay on one side of it (joint 2 spans 2.47-4.08 rad continuously),
so a 0.1 rad perturbation never crosses the branch cut.
"""
from __future__ import annotations

import numpy as np

from .schema import N_ARM, N_ROBOT, obs_slice


def perturb_and_relabel(obs: np.ndarray, action: np.ndarray,
                        sigma_arm: float = 0.05, sigma_hand: float = 0.0,
                        n_copies: int = 1, arm_only: bool = True,
                        rng: np.random.Generator | None = None,
                        jitter_qd: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return (obs_aug, action_aug): the originals plus `n_copies` noisy copies.

    The demonstrated ACTION is kept while the observed STATE is perturbed, which
    is what makes the pair a recovery label: the same target from a displaced
    state commands the motion that closes the displacement.

    Only the joint-position block (and optionally velocity) is perturbed. The
    object pose, forces and object-conditioning columns are left alone -- they
    are consequences of the world, not of the robot's configuration, and
    perturbing them would fabricate physics rather than coverage.

    `sigma_hand` defaults to 0: the hand's labels are not consistent enough
    across demonstrations for a synthetic finger target to be meaningful.
    """
    rng = rng or np.random.default_rng(0)
    sl_qa, sl_qh = obs_slice("q_arm"), obs_slice("q_hand")
    sl_da, sl_dh = obs_slice("qd_arm"), obs_slice("qd_hand")

    out_o, out_a = [obs], [action]
    for _ in range(max(0, n_copies)):
        o = obs.copy()
        n = o.shape[0]
        o[:, sl_qa] += rng.normal(0.0, sigma_arm, (n, N_ARM))
        if not arm_only and sigma_hand > 0:
            o[:, sl_qh] += rng.normal(0.0, sigma_hand, (n, N_ROBOT - N_ARM))
        if jitter_qd:
            # A displaced joint is also moving differently. Scale the velocity
            # noise by sigma/dt_typical would overstate it; a fraction of the
            # position noise keeps (q, qd) mutually plausible.
            o[:, sl_da] += rng.normal(0.0, sigma_arm, (n, N_ARM))
            if not arm_only and sigma_hand > 0:
                o[:, sl_dh] += rng.normal(0.0, sigma_hand,
                                          (n, N_ROBOT - N_ARM))
        out_o.append(o)
        out_a.append(action.copy())
    return np.concatenate(out_o), np.concatenate(out_a)


def augment_store(store, sigma_arm: float = 0.05, n_copies: int = 1,
                  sigma_hand: float = 0.0, seed: int = 0,
                  jitter_qd: bool = True) -> None:
    """Augment an EpisodeStore in place, per episode.

    Per-episode rather than pooled so the sequence datasets (BC-RNN, ACT) keep
    contiguous windows: a window must not straddle an original/perturbed
    boundary, and appending whole perturbed episodes preserves that.
    """
    rng = np.random.default_rng(seed)
    obs, act = [], []
    for o, a in zip(store.obs, store.act):
        oa, aa = perturb_and_relabel(
            o.astype(np.float64), a.astype(np.float64),
            sigma_arm=sigma_arm, sigma_hand=sigma_hand, n_copies=n_copies,
            arm_only=(sigma_hand <= 0), rng=rng, jitter_qd=jitter_qd)
        k = o.shape[0]
        # Keep episodes separate so windowing never crosses a seam.
        for i in range(n_copies + 1):
            obs.append(oa[i * k:(i + 1) * k].astype(np.float32))
            act.append(aa[i * k:(i + 1) * k].astype(np.float32))
    store.obs, store.act = obs, act
