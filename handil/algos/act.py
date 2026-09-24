"""ACT -- action-chunking transformer (the ALOHA baseline).

Two ideas carry the method, and both matter for this data:

  * Action chunking. Predicting a chunk of K actions from one observation and
    executing them open-loop cuts the effective horizon by K, which is the
    standard remedy for BC's compounding error. It also suits demos logged at
    ~42 Hz where consecutive actions are strongly correlated.
  * A CVAE latent. Teleop demonstrations of the same task differ between
    operators and attempts; a per-chunk latent absorbs that variation instead
    of forcing the decoder to average over it.

This is a state-only implementation: the original conditions on multiple
camera streams via a ResNet backbone, which we omit because the source logs
store no image arrays (pixels would have to be re-rendered offline). The
encoder/decoder transformer and the CVAE objective are otherwise faithful.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import Policy, mlp, register


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.shape[1]]


@register("act")
class ACT(Policy):
    def __init__(self, obs_dim: int, act_dim: int, d_model: int = 256,
                 nhead: int = 8, enc_layers: int = 4, dec_layers: int = 4,
                 dim_ff: int = 1024, horizon: int = 5, chunk: int = 16,
                 latent_dim: int = 32, kl_weight: float = 10.0,
                 dropout: float = 0.1):
        super().__init__(obs_dim, act_dim)
        self.horizon, self.chunk = horizon, chunk
        self.latent_dim, self.kl_weight = latent_dim, kl_weight

        self.obs_proj = nn.Linear(obs_dim, d_model)
        self.pos = PositionalEncoding(d_model)

        # CVAE encoder: sees the ground-truth chunk at training time only.
        self.act_proj = nn.Linear(act_dim, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True, norm_first=True)
        self.vae_encoder = nn.TransformerEncoder(enc_layer, 2)
        self.to_latent = nn.Linear(d_model, latent_dim * 2)

        self.latent_proj = nn.Linear(latent_dim, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout,
                                       batch_first=True, norm_first=True),
            enc_layers)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_ff, dropout,
                                       batch_first=True, norm_first=True),
            dec_layers)
        self.query = nn.Parameter(torch.zeros(1, chunk, d_model))
        nn.init.normal_(self.query, std=0.02)
        self.head = mlp([d_model, d_model, act_dim])

    def _encode_latent(self, obs_seq, act_chunk):
        b = obs_seq.shape[0]
        toks = torch.cat([self.cls.expand(b, -1, -1),
                          self.obs_proj(obs_seq),
                          self.act_proj(act_chunk)], dim=1)
        h = self.vae_encoder(self.pos(toks))[:, 0]
        mu, logvar = self.to_latent(h).chunk(2, dim=-1)
        return mu, logvar.clamp(-8.0, 8.0)

    def _decode(self, obs_seq, z):
        b = obs_seq.shape[0]
        mem = self.encoder(self.pos(
            torch.cat([self.latent_proj(z).unsqueeze(1),
                       self.obs_proj(obs_seq)], dim=1)))
        out = self.decoder(self.query.expand(b, -1, -1), mem)
        return self.head(out)

    def loss(self, batch):
        obs, act = batch
        if obs.ndim == 2:
            obs = obs.unsqueeze(1)
        if act.ndim == 2:
            act = act.unsqueeze(1)
        mu, logvar = self._encode_latent(obs, act)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        pred = self._decode(obs, z)
        # L1 as in the paper: less sensitive to the occasional large action
        # spike than MSE, which matters with torque-recovered labels.
        recon = nn.functional.l1_loss(pred, act)
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)).mean()
        return recon + self.kl_weight * kl

    def predict_chunk(self, obs_win):
        # At test time the latent prior mean (zero) is used, as in the paper.
        z = torch.zeros(obs_win.shape[0], self.latent_dim, device=obs_win.device)
        return self._decode(obs_win, z)
