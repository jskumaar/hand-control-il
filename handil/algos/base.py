"""Algorithm registry and the policy interface.

Every algorithm implements the same two-sided contract:

  * training:  `loss(batch) -> Tensor`
  * rollout:   `act(obs_window) -> np.ndarray(23,)`

`act` takes a window of raw (unnormalised) observations, newest last, so
sequence models and single-step models share one call site in the evaluator.
Action chunking is handled by the policy itself via an internal queue, which
keeps the chunk length an implementation detail rather than something the
rollout loop must know about.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn

_REGISTRY: dict[str, Callable[..., "Policy"]] = {}


def register(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        cls.algo_name = name
        return cls
    return deco


def build(name: str, **kw) -> "Policy":
    if name not in _REGISTRY:
        raise KeyError(f"unknown algo {name!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kw)


def available() -> list[str]:
    return sorted(_REGISTRY)


class Policy(nn.Module):
    """Base class. Subclasses set `horizon` and `chunk`."""

    algo_name = "base"
    horizon = 1     # observation history length required
    chunk = 1       # actions predicted per forward pass

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self._queue: list[np.ndarray] = []
        self.normalizer = None

    # -- training ---------------------------------------------------------
    def loss(self, batch) -> torch.Tensor:
        raise NotImplementedError

    # -- rollout ----------------------------------------------------------
    def predict_chunk(self, obs_win: torch.Tensor) -> torch.Tensor:
        """(1, horizon, obs_dim) normalised -> (1, chunk, act_dim) normalised."""
        raise NotImplementedError

    def reset(self):
        self._queue.clear()

    @torch.no_grad()
    def act(self, obs_window: np.ndarray) -> np.ndarray:
        """Raw obs window (T, obs_dim), newest last -> raw action (act_dim,)."""
        if self._queue:
            return self._queue.pop(0)
        self.eval()
        nz = self.normalizer
        win = np.asarray(obs_window, dtype=np.float64)
        if win.ndim == 1:
            win = win[None, :]
        win = win[-self.horizon:]
        if win.shape[0] < self.horizon:      # pad by repeating the oldest frame
            win = np.concatenate([np.repeat(win[:1], self.horizon - win.shape[0], 0), win])
        x = np.nan_to_num(nz.norm_obs(win)).astype(np.float32)
        t = torch.from_numpy(x).unsqueeze(0).to(next(self.parameters()).device)
        out = self.predict_chunk(t)[0].cpu().numpy()
        acts = [nz.denorm_act(a) for a in out]
        self._queue = list(acts[1:])
        return acts[0]


def mlp(sizes: list[int], act=nn.ReLU, out_act=None, dropout: float = 0.0):
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
            if dropout:
                layers.append(nn.Dropout(dropout))
    if out_act is not None:
        layers.append(out_act())
    return nn.Sequential(*layers)
