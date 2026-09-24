"""Smoke tests: every registered algorithm trains a step and acts."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from handil.algos import available, build
from handil.data.dataset import Normalizer
from handil.data.schema import ACT_DIM, OBS_DIM


@pytest.fixture
def normalizer():
    rng = np.random.default_rng(0)
    return Normalizer.fit(rng.normal(size=(64, OBS_DIM)),
                          rng.normal(size=(64, ACT_DIM)))


@pytest.mark.parametrize("name", available())
def test_loss_is_finite_and_backprops(name):
    p = build(name, obs_dim=OBS_DIM, act_dim=ACT_DIM)
    b = 8
    obs = torch.randn(b, max(p.horizon, 1), OBS_DIM)
    act = torch.randn(b, max(p.chunk, 1), ACT_DIM)
    loss = p.loss((obs, act))
    assert torch.isfinite(loss), f"{name} produced a non-finite loss"
    loss.backward()
    assert any(q.grad is not None and torch.isfinite(q.grad).all()
               for q in p.parameters())


@pytest.mark.parametrize("name", available())
def test_act_returns_one_raw_action_of_the_right_shape(name, normalizer):
    p = build(name, obs_dim=OBS_DIM, act_dim=ACT_DIM)
    p.normalizer = normalizer
    p.reset()
    win = np.random.randn(p.horizon, OBS_DIM)
    a = p.act(win)
    assert a.shape == (ACT_DIM,)
    assert np.isfinite(a).all()


@pytest.mark.parametrize("name", available())
def test_act_tolerates_a_short_history(name, normalizer):
    """Early rollout steps have fewer frames than the policy's horizon."""
    p = build(name, obs_dim=OBS_DIM, act_dim=ACT_DIM)
    p.normalizer = normalizer
    p.reset()
    a = p.act(np.random.randn(1, OBS_DIM))
    assert a.shape == (ACT_DIM,)


def test_chunked_policy_drains_its_queue_before_recomputing(normalizer):
    p = build("act", obs_dim=OBS_DIM, act_dim=ACT_DIM, chunk=4, horizon=2)
    p.normalizer = normalizer
    p.reset()
    win = np.random.randn(p.horizon, OBS_DIM)
    first = p.act(win)
    assert len(p._queue) == 3
    for _ in range(3):
        p.act(win)
    assert len(p._queue) == 0
    assert np.isfinite(first).all()


def test_reset_clears_the_queue(normalizer):
    p = build("act", obs_dim=OBS_DIM, act_dim=ACT_DIM, chunk=4, horizon=2)
    p.normalizer = normalizer
    p.act(np.random.randn(p.horizon, OBS_DIM))
    assert p._queue
    p.reset()
    assert not p._queue


def test_registry_has_the_expected_algorithms():
    assert {"bc_mlp", "bc_rnn", "act", "ibc"} <= set(available())


def test_unknown_algorithm_raises():
    with pytest.raises(KeyError):
        build("no_such_algo", obs_dim=OBS_DIM, act_dim=ACT_DIM)
