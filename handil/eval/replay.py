"""Replay recorded demonstrations through the rollout env.

This is the repo's calibration tool, not a policy evaluator. It answers one
question: does the environment, driven by the *demonstrated* actions, reproduce
the demonstrated outcome? That number is the ceiling any learned policy can be
measured against, and it separates "the policy is weak" from "the env, the
action labels, or the scene are wrong" -- a distinction that is otherwise very
easy to get wrong.

Replay is open-loop: the recorded action stream is applied verbatim, with no
feedback correction, so small tracking differences accumulate over an episode.
Expect it to be imperfect even when everything is correct.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.schema import N_ROBOT
from ..envs.rollout import (ARRIVAL_DWELL_S, ARRIVAL_SPEED_M_S, DROP_DWELL_S,
                            DWELL_S, LIFT_HEIGHT_M, PickPlaceEnv)


@dataclass
class ReplayResult:
    episode_id: str
    obj: str
    success: bool
    picked: bool
    max_height: float
    n_drops: int
    steps: int
    arm_err_p50: float
    arm_err_p95: float
    min_obj_in_palm: float


def replay_episode(env: PickPlaceEnv, actions: np.ndarray,
                   obs_ref: np.ndarray | None = None,
                   t: np.ndarray | None = None,
                   episode_id: str = "", obj: str = "",
                   extra_steps: int = 250) -> ReplayResult:
    """Drive `env` with a recorded action stream and score it.

    `t` is the episode's own timestamps: each action is held for the number of
    physics steps that matches the interval it was actually held for during
    collection. Without it the env's fixed control rate is used, which slowly
    desynchronises replay from a variable-rate recording.
    """
    obs = env.reset()
    n = len(actions)
    if t is not None and len(t) == n and n > 1:
        dt = np.diff(t, prepend=t[0] - float(np.median(np.diff(t))))
    else:
        dt = None
    base_substeps = env.n_substeps

    dwell_t0 = drop_t0 = arrive_t0 = None
    picked = success = False
    n_drops = 0
    max_h = -np.inf
    errs: list[float] = []
    min_oip = np.inf
    steps = 0

    for step in range(n + extra_steps):
        i = min(step, n - 1)
        if dt is not None:
            env.n_substeps = int(np.clip(
                round(dt[i] / env.model.opt.timestep), 1, 40))
        else:
            env.n_substeps = base_substeps

        obs = env.step(actions[i])
        steps = step + 1
        sim_t = float(env.data.time)

        if obs_ref is not None and i < len(obs_ref):
            errs.append(float(np.abs(obs[:7] - obs_ref[i, :7]).max()))
        oip = float(np.linalg.norm(obs[46:49]))
        if np.isfinite(oip):
            min_oip = min(min_oip, oip)

        h = env.object_height_above_rest()
        max_h = max(max_h, h)
        held = env.hand_touching(env.active_object)
        in_bowl = env.object_in_bowl(env.active_object)

        if not picked:
            if h > LIFT_HEIGHT_M and held:
                dwell_t0 = sim_t if dwell_t0 is None else dwell_t0
                if sim_t - dwell_t0 >= DWELL_S:
                    picked = True
            else:
                dwell_t0 = None

        if in_bowl and env.object_speed() < ARRIVAL_SPEED_M_S and not held:
            arrive_t0 = sim_t if arrive_t0 is None else arrive_t0
            if sim_t - arrive_t0 >= ARRIVAL_DWELL_S:
                success = True
                break
        else:
            arrive_t0 = None

        if picked and h <= LIFT_HEIGHT_M and not held and not in_bowl:
            drop_t0 = sim_t if drop_t0 is None else drop_t0
            if sim_t - drop_t0 >= DROP_DWELL_S:
                n_drops += 1
                picked = False
                dwell_t0 = drop_t0 = None
        else:
            drop_t0 = None

    env.n_substeps = base_substeps
    return ReplayResult(
        episode_id=episode_id, obj=obj, success=success, picked=picked,
        max_height=float(max_h), n_drops=n_drops, steps=steps,
        arm_err_p50=float(np.percentile(errs, 50)) if errs else float("nan"),
        arm_err_p95=float(np.percentile(errs, 95)) if errs else float("nan"),
        min_obj_in_palm=float(min_oip) if np.isfinite(min_oip) else float("nan"))
