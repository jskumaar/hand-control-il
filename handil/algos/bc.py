"""BC-MLP and BC-RNN -- the robomimic-style baselines."""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import Policy, mlp, register


@register("bc_mlp")
class BCMLP(Policy):
    """Feed-forward regression from one observation to one action.

    The weakest baseline on purpose: it cannot represent multimodality, so
    where teleop demos disagree about how to approach an object it averages
    them. That failure is informative -- it is the control against which the
    chunked and implicit policies are read.
    """

    horizon = 1
    chunk = 1

    def __init__(self, obs_dim: int, act_dim: int, hidden=(512, 512),
                 dropout: float = 0.0):
        super().__init__(obs_dim, act_dim)
        self.net = mlp([obs_dim, *hidden, act_dim], dropout=dropout)

    def forward(self, obs):
        return self.net(obs)

    def loss(self, batch):
        obs, act = batch
        if obs.ndim == 3:
            obs, act = obs[:, -1], act[:, 0]
        return nn.functional.mse_loss(self.net(obs), act)

    def predict_chunk(self, obs_win):
        return self.net(obs_win[:, -1]).unsqueeze(1)


@register("bc_rnn")
class BCRNN(Policy):
    """GRU over an observation window, predicting the next action.

    Gives the policy the temporal context a single frame lacks -- whether the
    hand is closing or opening looks identical in one frame of joint positions
    but not in ten.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 512,
                 layers: int = 2, horizon: int = 10, dropout: float = 0.0):
        super().__init__(obs_dim, act_dim)
        self.horizon = horizon
        self.chunk = 1
        self.rnn = nn.GRU(obs_dim, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = mlp([hidden, hidden, act_dim])

    def forward(self, obs_seq):
        out, _ = self.rnn(obs_seq)
        return self.head(out[:, -1])

    def loss(self, batch):
        obs, act = batch
        if obs.ndim == 2:
            obs = obs.unsqueeze(1)
        tgt = act[:, 0] if act.ndim == 3 else act
        return nn.functional.mse_loss(self(obs), tgt)

    def predict_chunk(self, obs_win):
        return self(obs_win).unsqueeze(1)
