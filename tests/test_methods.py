"""Tests for collection-method identification.

Every case here was observed in the real archive; see the module docstring of
handil/data/methods.py for the counts.
"""
from __future__ import annotations

from handil.data.methods import (PIPELINE_BASELINE, PIPELINE_CONTACT_AWARE,
                                 parse_dirname, resolve)


def test_directory_carries_the_retargeter_the_event_drops():
    """contact_aware_w_anyteleop and _w_vwj both log method=contact_aware_teleop."""
    a = resolve("contact_aware_w_anyteleop", "contact_aware_teleop")
    b = resolve("contact_aware_w_vwj", "contact_aware_teleop")
    assert a.pipeline == b.pipeline == PIPELINE_CONTACT_AWARE
    assert a.retargeter == "anyteleop"
    assert b.retargeter == "vwj"
    assert a.key != b.key            # the whole point: they stay distinguishable
    assert not a.conflict and not b.conflict


def test_event_wins_when_the_directory_is_stale():
    """One anyteleop_* directory holds trials whose events say dexpilot."""
    m = resolve("anyteleop", "dexpilot")
    assert m.pipeline == PIPELINE_BASELINE
    assert m.retargeter == "dexpilot"
    assert m.conflict


def test_ablation_suffix_is_split_out_and_poolable():
    m = resolve("contact_aware_w_anyteleop_s0p4", "contact_aware_teleop")
    assert m.ablation == "s0p4"
    assert m.retargeter == "anyteleop"
    assert m.key == "contact_aware+anyteleop@s0p4"
    assert m.base_key == "contact_aware+anyteleop"


def test_contact_aware_teleop_directory_means_in_house_dexpilot():
    m = resolve("contact_aware_teleop", "contact_aware_teleop")
    assert m.pipeline == PIPELINE_CONTACT_AWARE
    assert m.retargeter == "dexpilot"
    assert not m.conflict


def test_plain_baselines_round_trip():
    for r in ("dexpilot", "anyteleop", "vwj"):
        m = resolve(r, r)
        assert m.pipeline == PIPELINE_BASELINE
        assert m.retargeter == r
        assert not m.conflict


def test_parse_dirname_without_an_event():
    p, r, ab = parse_dirname("contact_aware_w_vwj_upstream_s0p8")
    assert (p, r, ab) == (PIPELINE_CONTACT_AWARE, "vwj_upstream", "s0p8")


def test_run_index_suffix_is_stripped():
    """`anyteleop_r01` must not resolve to a retargeter named 'anyteleop_r01' --
    that was its own private cell, pairing with nothing."""
    assert parse_dirname("anyteleop_r01") == (PIPELINE_BASELINE, "anyteleop", None)
    assert parse_dirname("contact_aware_w_anyteleop_r02") == (
        PIPELINE_CONTACT_AWARE, "anyteleop", None)
    # a run index may sit either side of the ablation tag
    assert parse_dirname("anyteleop_s0p4_r01") == (
        PIPELINE_BASELINE, "anyteleop", "s0p4")
    assert parse_dirname("anyteleop_r01_s0p4") == (
        PIPELINE_BASELINE, "anyteleop", "s0p4")


def test_new_runs_pair_with_each_other():
    a = resolve("anyteleop_r01", "anyteleop")
    b = resolve("contact_aware_w_anyteleop_r01", "contact_aware_teleop")
    assert a.key == "baseline+anyteleop"
    assert b.key == "contact_aware+anyteleop"
    assert a.base_key != b.base_key and a.retargeter == b.retargeter


def test_logged_mode_never_downgrades_the_retargeter():
    """`run_config.mode` is NORMALISED by hand-control before it is logged:
    `contact_aware_w_anyteleop` -> `contact_aware_teleop`, with the retargeter
    moved to a variable the event does not carry. Trusting it blindly relabels
    every *_w_anyteleop_r0* run as dexpilot, which silently emptied the
    contact_aware+anyteleop cell for runs r03-r09 while leaving 25 episodes
    sitting in the manifest under the wrong key."""
    from handil.data.logs import run_mode

    ca = [{"event": "run_config", "mode": "contact_aware_teleop"}]
    mode, from_log = run_mode(ca, "contact_aware_w_anyteleop_r03")
    assert mode == "contact_aware_w_anyteleop_r03" and not from_log
    assert resolve(mode, "contact_aware_teleop").retargeter == "anyteleop"

    # Same collapse on the baseline side: plain `anyteleop` logs as `dexpilot`
    # in run_config. The per-trial `method` field is NOT affected (it says
    # `anyteleop` for every baseline run in the archive), so only run_mode has
    # to defend against this one.
    bl = [{"event": "run_config", "mode": "dexpilot"}]
    mode, from_log = run_mode(bl, "anyteleop_r03")
    assert mode == "anyteleop_r03" and not from_log
    assert resolve(mode, "anyteleop").retargeter == "anyteleop"

    # But a genuinely in-house dexpilot run must still read from the log.
    mode, from_log = run_mode(ca, "contact_aware_teleop")
    assert mode == "contact_aware_teleop" and from_log
