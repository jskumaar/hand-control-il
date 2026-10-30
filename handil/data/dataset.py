"""Torch datasets over extracted episodes."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import ACT_DIM, OBS_DIM


@dataclass
class Normalizer:
    obs_mean: np.ndarray
    obs_std: np.ndarray
    act_mean: np.ndarray
    act_std: np.ndarray

    @staticmethod
    def fit(obs: np.ndarray, act: np.ndarray, eps: float = 1e-4) -> "Normalizer":
        return Normalizer(
            obs_mean=obs.mean(0), obs_std=np.maximum(obs.std(0), eps),
            act_mean=act.mean(0), act_std=np.maximum(act.std(0), eps))

    def norm_obs(self, o):
        return (o - self.obs_mean) / self.obs_std

    def norm_act(self, a):
        return (a - self.act_mean) / self.act_std

    def denorm_act(self, a):
        return a * self.act_std + self.act_mean

    def save(self, p: Path):
        np.savez(p, **{k: getattr(self, k) for k in
                       ("obs_mean", "obs_std", "act_mean", "act_std")})

    @staticmethod
    def load(p: Path) -> "Normalizer":
        d = np.load(p)
        return Normalizer(**{k: d[k] for k in d.files})


def episode_folds(dataset, n_folds: int, seed: int):
    """K-fold episode splits, so every episode is held out exactly once.

    Why this exists: with val_frac=0.1 the validation set is 1-2 EPISODES, so
    "validation loss" largely reports WHICH episode was held out. Measured on
    contact_aware+anyteleop, 8 seeds per episode count, the per-seed values span
    1-2 orders of magnitude (at 23 episodes: 0.0063 to 0.1540), and sd/mean is
    65-154% and does NOT shrink with more episodes. Averaging over folds removes
    that variance instead of averaging over it.
    """
    ep = getattr(dataset, "episode_of", None)
    if ep is None:
        raise ValueError("dataset cannot report episodes; use episode_split")
    ep = np.asarray(ep)
    uniq = np.random.default_rng(seed).permutation(np.unique(ep))
    chunks = np.array_split(uniq, min(n_folds, len(uniq)))
    for held in chunks:
        mask = np.isin(ep, held)
        yield list(np.flatnonzero(~mask)), list(np.flatnonzero(mask))


def episode_split(dataset, val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    """Train/val sample indices split by EPISODE, not by sample.

    A per-sample random_split puts near-duplicate frames on both sides: windows
    10 timesteps apart in the same episode are almost identical, so validation
    loss measures interpolation WITHIN a trajectory rather than generalisation
    to an unseen one, and reads optimistically. Holding out whole episodes is
    the comparison that matters for a policy that must run a fresh episode.

    Falls back to a sample split only if the dataset cannot report episodes.
    """
    ep = getattr(dataset, "episode_of", None)
    if ep is None:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(dataset))
        n_val = max(1, int(len(dataset) * val_frac))
        return list(idx[n_val:]), list(idx[:n_val])

    ep = np.asarray(ep)
    uniq = np.unique(ep)
    rng = np.random.default_rng(seed)
    uniq = rng.permutation(uniq)
    n_val_ep = max(1, int(round(len(uniq) * val_frac)))
    val_eps = set(uniq[:n_val_ep].tolist())
    mask = np.isin(ep, list(val_eps))
    return list(np.flatnonzero(~mask)), list(np.flatnonzero(mask))


def manifest_obs_dim(data_dir: Path) -> int:
    """Observation width of a dataset, from its manifest's opt-in extras.

    Lets a dataset built with --phase-onehot train without hardcoding a width.
    """
    from .schema import obs_dim

    m = json.loads((Path(data_dir) / "manifest.json").read_text())
    return obs_dim(tuple(m.get("obs_extras", ())))


def load_manifest(data_dir: Path) -> list[dict]:
    """Episodes from a manifest, refusing a stale on-disk schema.

    Shards are OBS_DIM-wide float arrays with no self-describing header, so a
    v1 shard loaded under v2 would silently mis-shape every batch. Fail loudly.
    """
    from .schema import SCHEMA_VERSION

    m = json.loads((data_dir / "manifest.json").read_text())
    got = m.get("schema_version", 1)
    if got != SCHEMA_VERSION:
        raise SystemExit(
            f"{data_dir} was extracted at schema_version={got} but this code is "
            f"at {SCHEMA_VERSION}. Re-extract with scripts/extract_data.py.")
    return m["episodes"]


def select(episodes: list[dict], pipeline=None, retargeter=None, method_key=None,
           obj=None, outcome=("success",), exclude_ablations: bool = True,
           exclude_conflicts: bool = False) -> list[dict]:
    """Filter episodes by collection method and quality."""
    def ok(e):
        if outcome and e["outcome"] not in outcome:
            return False
        if pipeline and e["pipeline"] != pipeline:
            return False
        if retargeter and e["retargeter"] != retargeter:
            return False
        if method_key and e["method_key"] != method_key:
            return False
        if obj and e["obj"] != obj:
            return False
        if exclude_ablations and e.get("ablation"):
            return False
        if exclude_conflicts and e.get("method_conflict"):
            return False
        return True
    return [e for e in episodes if ok(e)]


class EpisodeStore:
    """Loads selected episodes into memory once (the corpus is small)."""

    def __init__(self, data_dir: Path, episodes: list[dict]):
        self.data_dir = Path(data_dir)
        self.meta = episodes
        self.obs: list[np.ndarray] = []
        self.act: list[np.ndarray] = []
        for e in episodes:
            with np.load(self.data_dir / f"{e['episode_id']}.npz",
                         allow_pickle=True) as d:
                o, a = d["obs"].astype(np.float32), d["action"].astype(np.float32)
            # NaNs appear wherever a field was absent from the source schema
            # (e.g. obj_in_palm outside a trial). Zero-fill after normalisation
            # stats are computed, so they never pollute the mean/std.
            self.obs.append(o)
            self.act.append(a)

    @property
    def n_episodes(self) -> int:
        return len(self.obs)

    def flat(self) -> tuple[np.ndarray, np.ndarray]:
        return np.concatenate(self.obs), np.concatenate(self.act)

    def fit_normalizer(self) -> Normalizer:
        o, a = self.flat()
        m = np.isfinite(o).all(1)
        return Normalizer.fit(o[m] if m.any() else np.nan_to_num(o), a)


class TransitionDataset(Dataset):
    """Single (obs, action) pairs -- for BC-MLP and IBC."""

    def __init__(self, store: EpisodeStore, norm: Normalizer):
        o, a = store.flat()
        self.obs = torch.from_numpy(np.nan_to_num(norm.norm_obs(o)).astype(np.float32))
        self.act = torch.from_numpy(norm.norm_act(a).astype(np.float32))
        # Which episode each row came from, so the split can be per EPISODE.
        self.episode_of = np.concatenate(
            [np.full(x.shape[0], k, dtype=np.int64)
             for k, x in enumerate(store.obs)])

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, i):
        return self.obs[i], self.act[i]


class SequenceDataset(Dataset):
    """Windows of (obs_seq, action_chunk) -- for BC-RNN and ACT.

    `horizon` observations of history, `chunk` actions predicted from the last
    one. Episodes shorter than the window are skipped rather than padded, since
    a padded grasp is not a grasp.
    """

    def __init__(self, store: EpisodeStore, norm: Normalizer,
                 horizon: int = 10, chunk: int = 16):
        self.horizon, self.chunk = horizon, chunk
        self.obs, self.act, self.index = [], [], []
        for k, (o, a) in enumerate(zip(store.obs, store.act)):
            o = np.nan_to_num(norm.norm_obs(o)).astype(np.float32)
            a = norm.norm_act(a).astype(np.float32)
            self.obs.append(torch.from_numpy(o))
            self.act.append(torch.from_numpy(a))
            n = o.shape[0]
            if n < horizon + chunk:
                continue
            for t in range(horizon - 1, n - chunk):
                self.index.append((k, t))

    @property
    def episode_of(self) -> np.ndarray:
        """Episode index per sample -- for a per-episode train/val split."""
        return np.array([k for k, _ in self.index], dtype=np.int64)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        k, t = self.index[i]
        o = self.obs[k][t - self.horizon + 1: t + 1]
        a = self.act[k][t: t + self.chunk]
        return o, a
