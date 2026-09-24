"""IBC -- implicit behavioral cloning (Florence et al. 2021).

Instead of regressing obs -> action, learn an energy E(obs, action) and act by
minimising it. The point is representational: an explicit regressor must emit
one action and so averages conflicting demonstrations, while an energy surface
can hold several equally-low basins. With four retargeters producing
genuinely different solutions to the same grasp, that is the relevant failure.

Trained with the InfoNCE variant: the demonstrated action is the positive, and
negatives are sampled uniformly from the action bounds. Inference uses the
derivative-free optimiser from the paper -- sample, softmax-resample, shrink --
which avoids backprop through the energy at rollout time.

Included for completeness; it is well documented as finicky to train, and the
default hyper-parameters here are conservative rather than tuned.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import Policy, mlp, register


@register("ibc")
class IBC(Policy):
    horizon = 1
    chunk = 1

    def __init__(self, obs_dim: int, act_dim: int, hidden=(512, 512, 512),
                 n_negatives: int = 256, n_iters: int = 3, n_samples: int = 512,
                 shrink: float = 0.5, bound: float = 3.0, dropout: float = 0.0):
        super().__init__(obs_dim, act_dim)
        self.net = mlp([obs_dim + act_dim, *hidden, 1], dropout=dropout)
        self.n_negatives = n_negatives
        self.n_iters, self.n_samples, self.shrink = n_iters, n_samples, shrink
        # Actions are normalised, so a fixed symmetric bound covers the data.
        self.bound = bound

    def energy(self, obs, act):
        """obs (B, O), act (B, N, A) -> (B, N) energies."""
        b, n, _ = act.shape
        o = obs.unsqueeze(1).expand(-1, n, -1)
        return self.net(torch.cat([o, act], dim=-1)).squeeze(-1)

    def loss(self, batch):
        obs, act = batch
        if obs.ndim == 3:
            obs, act = obs[:, -1], act[:, 0]
        b = obs.shape[0]
        neg = (torch.rand(b, self.n_negatives, self.act_dim, device=obs.device)
               * 2 - 1) * self.bound
        cand = torch.cat([act.unsqueeze(1), neg], dim=1)      # positive at 0
        logits = -self.energy(obs, cand)
        target = torch.zeros(b, dtype=torch.long, device=obs.device)
        return nn.functional.cross_entropy(logits, target)

    @torch.no_grad()
    def predict_chunk(self, obs_win):
        obs = obs_win[:, -1]
        b = obs.shape[0]
        samples = (torch.rand(b, self.n_samples, self.act_dim, device=obs.device)
                   * 2 - 1) * self.bound
        scale = torch.full((b, 1, self.act_dim), self.bound, device=obs.device)
        for _ in range(self.n_iters):
            probs = torch.softmax(-self.energy(obs, samples), dim=-1)
            idx = torch.multinomial(probs, self.n_samples, replacement=True)
            samples = torch.gather(
                samples, 1, idx.unsqueeze(-1).expand(-1, -1, self.act_dim))
            scale = scale * self.shrink
            samples = samples + torch.randn_like(samples) * scale
            samples = samples.clamp(-self.bound, self.bound)
        best = torch.softmax(-self.energy(obs, samples), dim=-1).argmax(dim=-1)
        chosen = torch.gather(
            samples, 1, best[:, None, None].expand(-1, -1, self.act_dim))
        return chosen
