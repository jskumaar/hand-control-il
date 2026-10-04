"""Per-pipeline replay ceiling: are contact-aware demos more REPRODUCIBLE?

Policy-free. Drives the env with each demonstration's own recovered actions and
scores it, so it measures a property of the data rather than of any learner --
the ceiling a policy trained on that cell could reach.

Controlled the same way as the learning test: anyteleop only, both pipelines.
"""
import json
from pathlib import Path
import numpy as np
from handil.envs.rollout import PickPlaceEnv
from handil.envs.scene import build_scene
from handil.eval import replay_episode

DATA = Path("data/icra_all")
N = 24                       # per cell, keeps runtime sane
M = json.loads((DATA / "manifest.json").read_text())["episodes"]
scene = build_scene("robocasa")

out = {}
for pipe in ("baseline", "contact_aware"):
    eps = [e for e in M if e["retargeter"] == "anyteleop"
           and e["pipeline"] == pipe and e["n_objects"] == 4][:N]
    rows = []
    for e in eps:
        d = np.load(DATA / f"{e['episode_id']}.npz", allow_pickle=True)
        env = PickPlaceEnv(scene=scene, kp_hand=e["kp_hand"],
                           # By NAME, never e["obj_idx"]: that index is in the
                           # log trace's spawn frame, which differs from the
                           # scene's body order for 3 of the 4 ICRA objects.
                           active_object=e["obj"])
        r = replay_episode(env, d["action"], obs_ref=d["obs"], t=d["t"],
                           episode_id=e["episode_id"], obj=e["obj"])
        rows.append(r)
        print(f"  {pipe:14s} {r.obj:<22} "
              f"{'SUCCESS' if r.success else 'pick' if r.picked else '-':<8}"
              f" max_h={r.max_height:+.3f} armerr={r.arm_err_p95:.3f}", flush=True)
    out[pipe] = rows

print("\n=== replay ceiling by pipeline (anyteleop, 4-object scene) ===")
print(f"{'pipeline':16s} {'n':>3s} {'pick':>7s} {'success':>8s} {'armerr_p95':>11s}")
for pipe, rows in out.items():
    n = len(rows)
    print(f"{pipe:16s} {n:3d} {sum(r.picked for r in rows)/n:6.0%} "
          f"{sum(r.success for r in rows)/n:7.0%} "
          f"{np.median([r.arm_err_p95 for r in rows]):11.3f}")
Path("runs/pipe_replay.json").write_text(json.dumps(
    {p: [vars(r) for r in rows] for p, rows in out.items()}, indent=2))
