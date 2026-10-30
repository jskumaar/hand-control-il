"""Readers for hand-control run directories.

A run directory looks like:

    logs/<mode>_<timestamp>/
        pose_trace.npz                     full-state trace, ~42 Hz wall
        events.jsonl                       sparse trial transitions
        trial_<id>_<method>_<obj>_<outcome>.npz
        *_overview.mp4, *_wrist.mp4

Only pose_trace.npz + events.jsonl are used for learning: the trial npz files
are success-detection artifacts and (in baseline modes) omit q_robot entirely.

Schema variance is the defining problem here. Across the archive there are 8
distinct key-sets and qpos widths of 30 / 51 / 58, because the scene's object
count changed over time. Everything below is written to degrade gracefully:
a run missing a field is reported, not crashed on.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .schema import N_ROBOT

# Fields we can use if present; anything else in the npz is ignored.
REQUIRED = ("qpos", "t")
PREFERRED = ("qvel", "qfrc_applied", "qfrc_bias", "obj_in_palm",
             "obj_quat_in_palm", "norm_force", "tan_force", "t_wall", "phase")

_TRIAL_RE = re.compile(r"^trial_(\d+)_(.+?)_(obj_.+?)_(success|timeout|abandoned)\.npz$")


@dataclass
class TrialRecord:
    """One trial, as reconstructed from events.jsonl."""
    trial_id: int
    method: str
    obj: str
    outcome: str
    t_start: float           # sim time
    t_end: float
    t_start_wall: float
    t_end_wall: float
    props: dict = field(default_factory=dict)   # mass_kg, mu, shape, ...
    summary: dict = field(default_factory=dict)  # n_drops, duration_s, ...


@dataclass
class RunInfo:
    path: Path
    mode: str                # from the directory name
    keys: tuple[str, ...]
    n_rows: int
    nq: int
    trials: list[TrialRecord]
    missing: tuple[str, ...]

    @property
    def usable(self) -> bool:
        return self.n_rows > 0 and all(k in self.keys for k in REQUIRED)

    @property
    def has_torque(self) -> bool:
        return "qfrc_applied" in self.keys and "qfrc_bias" in self.keys


def mode_from_dirname(path: Path) -> str:
    """`contact_aware_w_anyteleop_20260915_164416` -> `contact_aware_w_anyteleop`."""
    name = path.name
    m = re.match(r"^(.*?)_\d{8}_\d{6}$", name)
    return m.group(1) if m else name


def find_runs(root: Path, include_obsolete: bool = False) -> list[Path]:
    """All run directories under `root` that contain a pose_trace.npz."""
    runs = []
    for p in sorted(root.rglob("pose_trace.npz")):
        d = p.parent
        parts = {q.name for q in d.parents}
        if not include_obsolete and any(
            "obsolete" in s or s == "divergence" for s in parts | {d.name}
        ):
            continue
        runs.append(d)
    return runs


def read_events(path: Path) -> list[dict]:
    """Parse events.jsonl, skipping malformed lines."""
    out = []
    if not path.exists():
        return out
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def trials_from_events(events: list[dict]) -> list[TrialRecord]:
    """Pair trial_start / trial_end events into TrialRecords.

    Trials without a matching end (the run was killed mid-trial) are dropped:
    they have no outcome, so they cannot be filtered for quality.
    """
    starts: dict[int, dict] = {}
    out: list[TrialRecord] = []
    for e in events:
        ev = e.get("event")
        tid = e.get("trial_id")
        if ev == "trial_start" and tid is not None:
            starts[tid] = e
        elif ev == "trial_end" and tid in starts:
            s = starts.pop(tid)
            props = {k: v for k, v in s.items()
                     if k not in ("event", "trial_id", "t", "t_wall",
                                  "method", "object")}
            summary = {k: v for k, v in e.items()
                       if k not in ("event", "trial_id", "t", "t_wall", "outcome")}
            out.append(TrialRecord(
                trial_id=int(tid),
                method=str(s.get("method", "unknown")),
                obj=str(s.get("object", "unknown")),
                outcome=str(e.get("outcome", "unknown")),
                t_start=float(s.get("t", np.nan)),
                t_end=float(e.get("t", np.nan)),
                t_start_wall=float(s.get("t_wall", np.nan)),
                t_end_wall=float(e.get("t_wall", np.nan)),
                props=props, summary=summary))
    return sorted(out, key=lambda r: r.trial_id)


PHASES = ("APPROACH", "PICK", "TRANSPORT", "PLACE")


def run_mode(events: list[dict], fallback: str) -> tuple[str, bool]:
    """The run's actual --mode, from the `run_config` event if hand-control
    logged one; otherwise the directory-name fallback.

    Why this matters: the per-trial `method` field is lossy (both
    contact_aware_w_anyteleop and contact_aware_w_vwj log
    `contact_aware_teleop`), so the retargeter could only be read off the
    DIRECTORY NAME -- which mis-resolves silently as soon as a run is named
    anything else. A real run named `cat_r01` resolved to
    `contact_aware+cat_r01`, i.e. its own private cell that would never pair
    with the baseline.

    Returns (mode, from_log). `from_log=False` means the fallback was used and
    the directory name had better contain the mode.

    CAUTION -- `run_config.mode` is the NORMALISED mode, not the CLI one.
    hand-control rewrites `args.mode` before logging it
    (kinova_leap_pick_place.py:561-575): `contact_aware_w_anyteleop` becomes
    `contact_aware_teleop` with the retargeter moved into a separate
    `_RETARGETER` variable that `run_config` does not carry, and plain
    `anyteleop` becomes `dexpilot` the same way. So the logged mode is
    retargeter-BLIND, exactly like the per-trial `method` field, and trusting
    it blindly relabels every `*_w_anyteleop_r0*` run as dexpilot -- which
    silently emptied the contact_aware+anyteleop cell for runs r03-r09.

    So: only prefer the logged mode when it is at least as specific as the
    directory name. The directory name encodes the full 2x2; the logged mode
    is authoritative about the PIPELINE but lossy about the retargeter.
    """
    from .methods import parse_dirname

    for e in events:
        if e.get("event") == "run_config" and e.get("mode"):
            logged = str(e["mode"])
            _lp, lr, _la = parse_dirname(logged)
            _fp, fr, _fa = parse_dirname(fallback)
            # If the directory names a retargeter and the logged mode does not
            # (it collapsed to the pipeline default), keep the directory's.
            if lr == "dexpilot" and fr not in ("dexpilot", "none"):
                return fallback, False
            return logged, True
    return fallback, False


def phase_per_row(events: list[dict], trial_id: int, t_wall: np.ndarray,
                  default: str = "APPROACH") -> np.ndarray:
    """Per-row phase for one trial, reconstructed from `phase_enter` events.

    Why not just read the trace's own `phase` column: it does NOT carry PICK for
    the baseline (anyteleop / vwj / dexpilot) modes, while `events.jsonl` does --
    measured on anyteleop_20260914_225108, whose trace has only
    {APPROACH, TRANSPORT, PLACE} but whose event stream has 4 `phase_enter/PICK`.
    So the annotation gap is in the per-row column, not in the source repo's
    logging, and reconstructing from the events closes it for every mode.

    Events are stamped with monotonic `t_wall`, which is also what episode rows
    are sliced on, so the two align without needing sim time (which resets).
    """
    out = np.full(len(t_wall), default, dtype="<U9")
    marks = sorted(
        ((float(e["t_wall"]), str(e["phase"])) for e in events
         if e.get("event") == "phase_enter" and e.get("trial_id") == trial_id
         and e.get("phase") in PHASES and e.get("t_wall") is not None),
        key=lambda kv: kv[0])
    for t_enter, phase in marks:
        out[t_wall >= t_enter] = phase       # later marks overwrite earlier ones
    return out


def inspect_run(run_dir: Path) -> RunInfo:
    """Cheap metadata read: opens the npz lazily, never materialises arrays."""
    npz_path = run_dir / "pose_trace.npz"
    with np.load(npz_path, allow_pickle=True) as d:
        keys = tuple(d.files)
        if "t" in keys:
            n_rows = int(d["t"].shape[0])
        elif "qpos" in keys:
            n_rows = int(d["qpos"].shape[0])
        else:
            n_rows = 0
        nq = int(d["qpos"].shape[1]) if "qpos" in keys and d["qpos"].ndim > 1 else 0
    trials = trials_from_events(read_events(run_dir / "events.jsonl"))
    missing = tuple(k for k in REQUIRED + PREFERRED if k not in keys)
    return RunInfo(path=run_dir, mode=mode_from_dirname(run_dir), keys=keys,
                   n_rows=n_rows, nq=nq, trials=trials, missing=missing)


def load_trace(run_dir: Path) -> dict[str, np.ndarray]:
    """Load pose_trace.npz into a plain dict of arrays."""
    with np.load(run_dir / "pose_trace.npz", allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def n_objects(nq: int) -> int:
    """Object count implied by the qpos width: 23 robot + 7 per freejoint."""
    rest = nq - N_ROBOT
    if rest < 0 or rest % 7:
        return 0
    return rest // 7


def pad_forces(arr: np.ndarray, n: int = 4) -> np.ndarray:
    """Normalise the (N,2) vs (N,4) force width to a fixed 4 columns.

    Pre-fix runs logged only thumb+index (kinova_leap_pick_place.py:6041); the
    finger order is ('thumb','index','middle','ring'), so left-padding is
    wrong and the missing middle/ring columns are filled with zeros at the end.
    """
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[1] >= n:
        return arr[:, :n]
    out = np.zeros((arr.shape[0], n), dtype=arr.dtype)
    out[:, : arr.shape[1]] = arr
    return out
