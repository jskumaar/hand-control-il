"""Training loop shared by every algorithm."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .algos import build
from .data.dataset import (EpisodeStore, Normalizer, SequenceDataset,
                           TransitionDataset, episode_split, load_manifest,
                           manifest_obs_dim, select)
from .data.schema import ACT_DIM, OBS_DIM


@dataclass
class TrainConfig:
    algo: str = "bc_mlp"
    data_dir: str = "data/episodes"
    out_dir: str = "runs"
    # data selection -- the experimental variable
    pipeline: str | None = None
    retargeter: str | None = None
    method_key: str | None = None
    obj: str | None = None
    outcome: tuple[str, ...] = ("success",)
    exclude_ablations: bool = True
    max_episodes: int | None = None
    # optimisation
    epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 1e-4
    val_frac: float = 0.1
    seed: int = 0
    device: str = "auto"
    num_workers: int = 0
    horizon: int = 10
    chunk: int = 16
    algo_kwargs: dict = None
    # DART-style noise injection (handil/data/dart.py). 0 copies = off.
    dart_copies: int = 0
    dart_sigma_arm: float = 0.05
    dart_sigma_hand: float = 0.0

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def device_note(self) -> str:
        """Human-readable device line, loud about an unintended CPU fallback.

        The fallback is silent and expensive -- a torch built for a newer CUDA
        than the installed driver reports is_available() == False and training
        runs ~57x slower with only a UserWarning. So say so at startup.
        """
        dev = self.resolved_device()
        if dev != "cpu":
            name = torch.cuda.get_device_name(0)
            return f"{dev} ({name})"
        if self.device == "auto" and torch.cuda.device_count() > 0:
            return ("cpu  [!] GPUs are present but torch cannot use them -- "
                    f"this torch is built for CUDA {torch.version.cuda}; "
                    "reinstall with `uv sync --extra cuda`")
        return "cpu"


def make_datasets(cfg: TrainConfig):
    data_dir = Path(cfg.data_dir)
    eps = select(load_manifest(data_dir), pipeline=cfg.pipeline,
                 retargeter=cfg.retargeter, method_key=cfg.method_key,
                 obj=cfg.obj, outcome=tuple(cfg.outcome),
                 exclude_ablations=cfg.exclude_ablations)
    if not eps:
        raise SystemExit("no episodes matched the selection")
    if cfg.max_episodes:
        # Deterministic subsample, so a data-scaling sweep is reproducible.
        rng = np.random.default_rng(cfg.seed)
        idx = rng.permutation(len(eps))[: cfg.max_episodes]
        eps = [eps[i] for i in sorted(idx)]

    store = EpisodeStore(data_dir, eps)
    if cfg.dart_copies > 0:
        from .data.dart import augment_store
        n_before = sum(o.shape[0] for o in store.obs)
        augment_store(store, sigma_arm=cfg.dart_sigma_arm,
                      sigma_hand=cfg.dart_sigma_hand,
                      n_copies=cfg.dart_copies, seed=cfg.seed)
        n_after = sum(o.shape[0] for o in store.obs)
        print(f"  DART: {n_before} -> {n_after} rows "
              f"({cfg.dart_copies} copies, sigma_arm={cfg.dart_sigma_arm})")
    # Normaliser is fitted AFTER augmentation so its std reflects the widened
    # state distribution the policy will actually be trained on.
    norm = store.fit_normalizer()

    # Width comes from the dataset, not a constant: opt-in blocks such as
    # phase_onehot widen it, and the checkpoint records what it was trained at.
    dim = manifest_obs_dim(data_dir)
    probe = build(cfg.algo, obs_dim=dim, act_dim=ACT_DIM,
                  **(cfg.algo_kwargs or {}))
    if probe.horizon > 1 or probe.chunk > 1:
        ds = SequenceDataset(store, norm, horizon=probe.horizon, chunk=probe.chunk)
    else:
        ds = TransitionDataset(store, norm)
    if len(ds) == 0:
        raise SystemExit("dataset is empty after windowing; lower horizon/chunk")
    return ds, norm, eps, probe, dim


def train(cfg: TrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = cfg.resolved_device()

    print(f"  device: {cfg.device_note()}")
    ds, norm, eps, policy, dim = make_datasets(cfg)
    # Split by EPISODE: a per-sample split leaks near-duplicate frames into
    # validation (see dataset.episode_split).
    i_tr, i_va = episode_split(ds, cfg.val_frac, cfg.seed)
    tr, va = Subset(ds, i_tr), Subset(ds, i_va)
    dl_tr = DataLoader(tr, batch_size=cfg.batch_size, shuffle=True,
                       num_workers=cfg.num_workers, drop_last=True)
    dl_va = DataLoader(va, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    policy = policy.to(dev)
    policy.normalizer = norm
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    norm.save(out / "normalizer.npz")
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
    (out / "episodes.json").write_text(json.dumps(eps, indent=2))

    hist, best = [], float("inf")
    t0 = time.time()
    for ep in range(cfg.epochs):
        policy.train()
        tot = n = 0
        for batch in dl_tr:
            batch = [b.to(dev) for b in batch]
            loss = policy.loss(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            tot += loss.detach().item() * len(batch[0])
            n += len(batch[0])
        tr_loss = tot / max(n, 1)

        policy.eval()
        tot = n = 0
        with torch.no_grad():
            for batch in dl_va:
                batch = [b.to(dev) for b in batch]
                tot += policy.loss(batch).detach().item() * len(batch[0])
                n += len(batch[0])
        va_loss = tot / max(n, 1)
        sched.step()
        hist.append({"epoch": ep, "train": tr_loss, "val": va_loss})

        if va_loss < best:
            best = va_loss
            torch.save({"state_dict": policy.state_dict(), "algo": cfg.algo,
                        "algo_kwargs": cfg.algo_kwargs or {},
                        "obs_dim": dim, "act_dim": ACT_DIM},
                       out / "best.pt")
        if ep % max(1, cfg.epochs // 20) == 0 or ep == cfg.epochs - 1:
            print(f"  epoch {ep:4d}  train {tr_loss:.5f}  val {va_loss:.5f}")

    summary = {"algo": cfg.algo, "n_episodes": len(eps), "n_samples": len(ds),
               "best_val": best, "train_s": time.time() - t0,
               "device": dev, "history": hist}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_policy(run_dir: Path, device: str = "cpu", expect_dim: int | None = None):
    ck = torch.load(Path(run_dir) / "best.pt", map_location=device,
                    weights_only=False)
    # The checkpoint records the width it was trained at, which may include
    # opt-in blocks (e.g. phase_onehot). Build at that width, and only complain
    # if it disagrees with what the caller is about to feed it.
    got = ck.get("obs_dim")
    want = expect_dim if expect_dim is not None else got
    if got != want:
        raise SystemExit(
            f"{run_dir} was trained with obs_dim={got} but will be fed "
            f"{want}-wide observations. Retrain, or point at the matching dataset.")
    p = build(ck["algo"], obs_dim=ck["obs_dim"], act_dim=ck["act_dim"],
              **ck.get("algo_kwargs", {}))
    p.load_state_dict(ck["state_dict"])
    p.normalizer = Normalizer.load(Path(run_dir) / "normalizer.npz")
    return p.to(device).eval()
