"""Closed-loop evaluation of a trained policy."""
from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..data.actions import kp_hand_for_mode
from ..envs.rollout import PickPlaceEnv, rollout
from ..envs.scene import build_scene


@dataclass
class EvalSummary:
    n_episodes: int
    success_rate: float
    pick_rate: float
    mean_steps: float
    mean_max_height: float
    mean_drops: float
    outcomes: dict


class WindowedPolicy:
    """Adapts a Policy to the env's single-observation callback.

    Keeps the observation history the policy needs and resets the internal
    action-chunk queue between episodes.
    """

    def __init__(self, policy, horizon: int):
        self.policy = policy
        self.buf: deque = deque(maxlen=max(1, horizon))

    def reset(self):
        self.buf.clear()
        self.policy.reset()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        self.buf.append(np.asarray(obs, float))
        return self.policy.act(np.stack(self.buf))


def evaluate(policy, n_episodes: int = 20, max_steps: int = 600,
             kp_hand: float = 2.0, scene_name: str = "pick_place",
             root=None, seed: int = 0, control_hz: float = 42.0,
             verbose: bool = True) -> tuple[EvalSummary, list[dict]]:
    """Roll the policy out repeatedly and score with the source's criteria.

    The scene is deterministic (the source repo hardcodes no-randomize), so
    repeated rollouts of a deterministic policy would be identical. We perturb
    the initial object pose slightly per episode to get a meaningful success
    rate rather than one sample repeated n times.
    """
    scene = build_scene(scene_name, root=root)
    env = PickPlaceEnv(scene=scene, kp_hand=kp_hand, control_hz=control_hz)
    wp = WindowedPolicy(policy, getattr(policy, "horizon", 1))
    rng = np.random.default_rng(seed)

    rows, n_succ, n_pick = [], 0, 0
    from collections import Counter
    outcomes = Counter()

    for i in range(n_episodes):
        env.reset()
        if scene.object_qadr:
            qadr = scene.object_qadr[0]
            jitter = rng.uniform(-0.02, 0.02, size=2)   # +-2 cm in x, y
            env.data.qpos[qadr:qadr + 2] += jitter
            env.mj.mj_forward(env.model, env.data)
            env._rest_z = [float(env.data.xpos[
                env.mj.mj_name2id(env.model, env.mj.mjtObj.mjOBJ_BODY, n)][2])
                for n in scene.object_bodies]
        wp.reset()
        r = rollout(env, wp, max_steps=max_steps)
        n_succ += r.success
        n_pick += r.picked
        outcomes[r.outcome] += 1
        rows.append({"episode": i, **{k: v for k, v in asdict(r).items()
                                      if k not in ("obs", "action")}})
        if verbose:
            print(f"  ep {i:3d}  {r.outcome:8s} steps={r.steps:4d} "
                  f"max_h={r.max_height:+.3f} picked={r.picked}")

    s = EvalSummary(
        n_episodes=n_episodes,
        success_rate=n_succ / max(n_episodes, 1),
        pick_rate=n_pick / max(n_episodes, 1),
        mean_steps=float(np.mean([r["steps"] for r in rows])),
        mean_max_height=float(np.mean([r["max_height"] for r in rows])),
        mean_drops=float(np.mean([r["n_drops"] for r in rows])),
        outcomes=dict(outcomes))
    return s, rows
