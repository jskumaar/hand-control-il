"""Identifying which data-collection method produced a trial.

This is the axis the whole study compares along, and it is genuinely awkward
in the source logs, so it gets its own module.

Two independent labels exist and NEITHER alone is sufficient:

  * The run directory name, e.g. `contact_aware_w_anyteleop_20260915_164416`.
    Encodes the full 2x2 factorial -- pipeline x retargeter -- plus ablation
    suffixes (`_s0p4` = solref 0.004 etc). But it is a human-chosen directory
    name and is occasionally wrong: one `anyteleop_*` directory in the archive
    contains trials whose events say `dexpilot`.

  * The `method` field in each trial_start event. Written by the trial logger
    from the live mode, so it is authoritative about the *pipeline* -- but it
    is lossy: `contact_aware_w_anyteleop` and `contact_aware_w_vwj` BOTH log
    `contact_aware_teleop`, discarding which retargeter was in use.

Verified against the archive (185 runs, 462 trials):

    dir=contact_aware_w_anyteleop   ->  event_method=contact_aware_teleop  (87)
    dir=contact_aware_w_vwj         ->  event_method=contact_aware_teleop (107)
    dir=anyteleop                   ->  event_method=dexpilot              (20)

So we decompose into two orthogonal factors and resolve each from whichever
source is reliable for it: the pipeline from the event, the retargeter from
the directory. Disagreements are surfaced, never silently resolved.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Pipeline: does the contact-aware grasp recommender run, or is it raw teleop?
PIPELINE_BASELINE = "baseline"
PIPELINE_CONTACT_AWARE = "contact_aware"
PIPELINE_AUTONOMOUS = "autonomous"

# Retargeter: what maps the operator's hand onto the LEAP hand.
RETARGETERS = ("dexpilot", "anyteleop", "vwj", "vwj_upstream", "none")

_SUFFIX_RE = re.compile(r"_(s0p\d+)$")
# A run-index suffix such as `_r01`, used to keep repeated sessions of the same
# condition in separate directories. It carries no experimental meaning and must
# be stripped before the retargeter is read, or `anyteleop_r01` resolves to a
# retargeter literally named "anyteleop_r01" -- its own private cell that would
# never pair with anything.
_RUNIDX_RE = re.compile(r"_r\d+$")


@dataclass(frozen=True)
class Method:
    """A collection method, decomposed into comparable factors."""
    pipeline: str
    retargeter: str
    ablation: str | None = None   # e.g. "s0p4" (solref stiffness sweep)
    conflict: bool = False        # directory and event disagreed

    @property
    def key(self) -> str:
        """Stable grouping key, e.g. `contact_aware+anyteleop` or
        `baseline+dexpilot@s0p4`."""
        base = f"{self.pipeline}+{self.retargeter}"
        return f"{base}@{self.ablation}" if self.ablation else base

    @property
    def base_key(self) -> str:
        """Grouping key ignoring the ablation sweep."""
        return f"{self.pipeline}+{self.retargeter}"

    def __str__(self) -> str:
        return self.key


def parse_dirname(mode: str) -> tuple[str, str, str | None]:
    """Decompose a run-directory mode string into (pipeline, retargeter, ablation)."""
    # Strip a run index first (it may sit outside the ablation suffix, e.g.
    # `anyteleop_s0p4_r01`), then the ablation tag.
    m = _RUNIDX_RE.search(mode)
    if m:
        mode = mode[: m.start()]
    ablation = None
    m = _SUFFIX_RE.search(mode)
    if m:
        ablation = m.group(1)
        mode = mode[: m.start()]
    # A run index can also precede the ablation tag; strip once more.
    m = _RUNIDX_RE.search(mode)
    if m:
        mode = mode[: m.start()]

    if mode.startswith("contact_aware_w_"):
        return PIPELINE_CONTACT_AWARE, mode[len("contact_aware_w_"):], ablation
    if mode in ("contact_aware_autonomous", "rrt"):
        return PIPELINE_AUTONOMOUS, "none", ablation
    if mode == "contact_aware_teleop":
        # The contact-aware pipeline with the in-house DexPilot retargeter.
        return PIPELINE_CONTACT_AWARE, "dexpilot", ablation
    if mode in RETARGETERS:
        return PIPELINE_BASELINE, mode, ablation
    return PIPELINE_BASELINE, mode, ablation


def pipeline_from_event(method: str) -> str:
    """Pipeline implied by a trial_start `method` field (authoritative)."""
    if method.startswith("contact_aware_autonomous") or method == "rrt":
        return PIPELINE_AUTONOMOUS
    if method.startswith("contact_aware"):
        return PIPELINE_CONTACT_AWARE
    return PIPELINE_BASELINE


def resolve(dir_mode: str, event_method: str | None) -> Method:
    """Combine both labels into one Method.

    Pipeline comes from the event when available (the directory name can be
    stale); retargeter comes from the directory (the event drops it). If the
    two imply different pipelines, the event wins and `conflict` is set so the
    extraction report can list the affected runs.
    """
    pipeline, retargeter, ablation = parse_dirname(dir_mode)
    conflict = False

    if event_method:
        ev_pipeline = pipeline_from_event(event_method)
        if ev_pipeline != pipeline:
            conflict = True
            pipeline = ev_pipeline
            if ev_pipeline == PIPELINE_BASELINE:
                # Directory said contact-aware but the run was raw teleop; the
                # event's method name then also names the retargeter.
                if event_method in RETARGETERS:
                    retargeter = event_method
            elif ev_pipeline == PIPELINE_AUTONOMOUS:
                retargeter = "none"
        elif (pipeline == PIPELINE_BASELINE and event_method in RETARGETERS
                and event_method != retargeter):
            # Same pipeline, but the directory misnames the retargeter (the
            # anyteleop/ dir holding dexpilot trials). Trust the event.
            #
            # NOTE this reads the per-trial `method` field, which DOES name the
            # retargeter correctly for every baseline run in the archive. It is
            # `run_config.mode` -- a different field -- that collapses
            # `anyteleop` to `dexpilot`; that one is handled in
            # logs.run_mode(), not here.
            conflict = True
            retargeter = event_method

    return Method(pipeline=pipeline, retargeter=retargeter,
                  ablation=ablation, conflict=conflict)
