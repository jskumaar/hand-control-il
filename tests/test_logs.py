"""Tests for the log readers, especially the schema-variance handling."""
from __future__ import annotations

import json

import numpy as np

from handil.data.logs import (mode_from_dirname, n_objects, pad_forces,
                              read_events, trials_from_events)


def test_mode_strips_the_timestamp():
    assert mode_from_dirname_p("contact_aware_w_anyteleop_20260915_164416") \
        == "contact_aware_w_anyteleop"
    assert mode_from_dirname_p("dexpilot_20260728_011543") == "dexpilot"
    assert mode_from_dirname_p("no_timestamp_here") == "no_timestamp_here"


def mode_from_dirname_p(name):
    from pathlib import Path
    return mode_from_dirname(Path(name))


def test_n_objects_from_qpos_width():
    assert n_objects(30) == 1      # single-object legacy scenes
    assert n_objects(51) == 4      # the 4-object ICRA scenes
    assert n_objects(58) == 5
    assert n_objects(23) == 0
    assert n_objects(25) == 0      # not a whole number of freejoints


def test_pad_forces_keeps_finger_order():
    """Pre-fix runs logged thumb+index only; middle/ring pad at the END."""
    two = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = pad_forces(two)
    assert out.shape == (2, 4)
    np.testing.assert_allclose(out[:, :2], two)
    np.testing.assert_allclose(out[:, 2:], 0.0)

    four = np.arange(8, dtype=float).reshape(2, 4)
    np.testing.assert_allclose(pad_forces(four), four)


def test_trials_pair_start_and_end_and_drop_unfinished(tmp_path):
    p = tmp_path / "events.jsonl"
    rows = [
        {"event": "trial_start", "trial_id": 1, "t": 0.0, "t_wall": 100.0,
         "method": "anyteleop", "object": "obj_017_orange", "mass_kg": 0.1},
        {"event": "phase_enter", "trial_id": 1, "t": 1.0, "t_wall": 101.0},
        {"event": "trial_end", "trial_id": 1, "t": 5.0, "t_wall": 105.0,
         "outcome": "success", "n_drops": 0},
        {"event": "trial_start", "trial_id": 2, "t": 6.0, "t_wall": 106.0,
         "method": "anyteleop", "object": "obj_025_mug"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))

    trials = trials_from_events(read_events(p))
    assert len(trials) == 1            # trial 2 never ended
    t = trials[0]
    assert t.outcome == "success"
    assert t.obj == "obj_017_orange"
    assert t.t_start_wall == 100.0 and t.t_end_wall == 105.0
    assert t.props["mass_kg"] == 0.1   # ground-truth physics kept
    assert t.summary["n_drops"] == 0


def test_read_events_skips_malformed_lines(tmp_path):
    p = tmp_path / "events.jsonl"
    p.write_text('{"event": "a"}\nnot json\n\n{"event": "b"}\n')
    assert [e["event"] for e in read_events(p)] == ["a", "b"]
