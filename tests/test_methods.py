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
