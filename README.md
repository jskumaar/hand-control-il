# handil — imitation learning from `hand-control` teleoperation logs

Learn manipulation policies from the demonstrations recorded by
[`hand-control`](../hand-control), and compare what the **data-collection
method** does to the policy you get out.

The experimental variable is the demonstration source, not the architecture.
`hand-control` records the same pick-and-place task under several collection
methods — a contact-aware grasp pipeline versus raw teleop, crossed with four
retargeters — and this repo asks which of them produces data a policy can
actually learn from.

Kinova Gen3 (7 DOF) + LEAP hand (16 DOF) = 23 DOF, MuJoCo, 500 Hz physics.

---

## Quick start

```bash
uv sync --extra dev                       # torch, mujoco, pytest
export HANDCONTROL_ROOT=~/sureshkj/hand-control

# 1. Build the episode dataset from the raw run logs (~2 min)
uv run python scripts/extract_data.py \
    --logs $HANDCONTROL_ROOT/logs --out data/episodes --outcomes success

# 2. Train one policy on one collection method
uv run python scripts/train_policy.py \
    --algo bc_rnn --retargeter anyteleop --pipeline contact_aware \
    --epochs 100 --out runs/ca_anyteleop_bcrnn

# 3. Score it closed-loop in the real MuJoCo scene
uv run python scripts/evaluate_policy.py --run runs/ca_anyteleop_bcrnn --episodes 20

# 4. Or run the whole comparison matrix (methods x algorithms)
uv run python scripts/run_comparison.py --algos bc_mlp bc_rnn act --episodes 20
```

The source repo is **read-only** to this one: its `logs/`, `models/` and YCB
assets are referenced by path, never copied, so the scene stays single-sourced.

---

## The two hard parts

Most of the engineering here is in making the source data learnable at all.
Both problems below were found by validating against the real archive, and
both are easy to get silently wrong.

### 1. Actions are not logged

`hand-control` never records `_dp_target`, the 23-vector of joint position
targets that is the semantic action. It records the *consequence*: the torque
written to `data.qfrc_applied`. The controller is a P law with gravity
compensation, so the action is recoverable exactly:

```
target = qpos[:23] + (qfrc_applied[:23] − qfrc_bias[:23]) / Kp
```

This is exact, not approximate, because the arm's Kd term is **identically
zero by construction** — the drive loop sets `data.qvel[:7] = _qdot_arm` on
the line before `Kd * (_qdot_arm − data.qvel[:7])` is evaluated. Real damping
lives in `model.dof_damping`, not in the commanded torque.

The gains are computed from the model, never fitted: the arm's are
`diag(M)[:7] * 30²` evaluated at the wrist-down home pose, and the hand's are
a scalar chosen by pipeline (0.8 contact-aware, 2.0 baseline).

> An earlier version inferred gains from trajectory smoothness. Checked against
> the analytic value it was wrong by up to 1000×. Don't reintroduce it.

**Validation.** Replaying recovered actions open-loop through a fresh
simulation — no feedback, no correction — tracks the logged trajectory to
**0.003 rad median / 0.09 rad p95 over 1500 steps**. The labels reproduce the
demonstrated motion.

**One exception.** During the contact-aware `GRASP` phase the pipeline adds
internal-force and slip-correction torques on top of the PD term, so
`qfrc_applied` is no longer `Kp·err` and inverting it gives unphysical finger
targets (up to 20 rad, versus 0.017 rad during `REACH`). Those rows have their
*finger* columns replaced by `qpos[t+1]`; the arm is unaffected. The manifest
reports `grasp_repair_frac` per episode — currently ~26% of rows across 120 of
235 episodes.

### 2. The method label is lossy

Identifying which collection method produced a trial needs **both** available
labels, because neither is correct alone:

| Source | Carries | Fails |
|---|---|---|
| `events.jsonl` `method` | the pipeline, authoritatively | collapses `contact_aware_w_anyteleop` and `_w_vwj` into one label, discarding the retargeter |
| run directory name | the full factorial + ablation suffix | occasionally stale — one `anyteleop_*` dir holds trials whose events say `dexpilot` |

So a method is decomposed into orthogonal factors — **pipeline** from the
event, **retargeter** from the directory — and disagreements set a `conflict`
flag rather than being silently resolved. 15 of 235 episodes currently
conflict. Group with `Method.key` (`contact_aware+anyteleop@s0p4`) or
`base_key` to pool the solref ablations.

---

## What's in the dataset

235 successful episodes from 195 runs (the extractor reads all of them; none
fail, and no episode needs the `next_qpos` action fallback).

| Collection method | Episodes | Steps |
|---|---:|---:|
| `contact_aware+vwj` | 38 | 29,643 |
| `baseline+dexpilot` | 38 | 71,919 |
| `contact_aware+anyteleop` | 23 | 24,117 |
| `baseline+anyteleop` | 16 | 19,758 |
| `baseline+vwj` | 13 | 17,421 |
| `contact_aware+dexpilot` | 6 | 3,998 |
| *(+ solref ablations `@s0p4/6/8`)* | 85 | 110,972 |

Filter on outcome: roughly half of all recorded trials are `abandoned`
(the operator switched target mid-attempt), and those are partial
demonstrations. Extraction keeps `success` only by default.

### Observation (68-d, state only)

`q_arm(7) · q_hand(16) · qd_arm(7) · qd_hand(16) · obj_in_palm(3) ·
obj_quat_in_palm(4) · palm_pos(3) · palm_quat(4) · norm_force(4) · tan_force(4)`

The object pose is in the **palm frame** deliberately: a world-frame trace
cannot separate "the hand carried the object" from "the object moved inside
the hand", while the palm frame cancels rigid hand motion so any residual is
grasp slip.

### Action (23-d)

Target joint position in radians, clipped to joint limits — the same space the
rollout env applies, so labels and executed actions live in one space.

### Images

Not included. The source logs store **no image arrays** — `SceneRecorder`
renders straight to MP4 and discards the pixels. Since every run is recorded
against a fixed, non-randomized scene with full `qpos` logged, frames are
reconstructible offline by replaying state through `mj.Renderer`. That is the
path to a vision policy; it needs a render-cache pass first.

---

## Evaluation

Policies are scored by **closed-loop rollout in the real scene**, not by
validation loss — offline BC loss correlates poorly with task success.

`handil/envs/` rebuilds the scene the way the source does (MjSpec surgery,
`base_link` re-posed onto the table and yawed 90°, hand self-collision off, YCB
objects attached from `scene_objects.json` and dropped onto the table), then
applies the same P-torque control with no torque clamping. The recovered arm
gains match the source exactly: `[509.3, 675.5, 254.3, 324.0, 65.9, 65.0, 3.1]`.

Success uses the source's own criteria from `trial_logger.py`: a lift of 1 cm
held for 1 s to confirm a pick, then contact-based containment in the bin
(contact with `bowl_base`, walls excluded, 0.2 s flicker latch) while settled
and released for 0.5 s.

### One known train/rollout mismatch

The demos were collected with **arm velocity injection** — the drive loop
writes `data.qvel[:7] = _qdot_arm` each substep so a teleoperator's jog rate
rides open-loop. A policy emits position targets, not rates, so injecting a
velocity it never commanded would fight its own target; the rollout env leaves
it off. This is a genuine domain gap between the demonstrations and the
rollout, stated here rather than papered over. `PickPlaceEnv(velocity_injection=True)`
exists to A/B it.

---

## Algorithms

| Name | What it is | Why it's here |
|---|---|---|
| `bc_mlp` | one obs → one action | the control: cannot represent multimodality, so it averages conflicting demos |
| `bc_rnn` | GRU over a 10-frame window | temporal context — a closing and an opening hand look identical in one frame |
| `act` | action-chunking transformer + CVAE latent | chunking cuts the effective horizon; the latent absorbs operator variation |
| `ibc` | implicit energy-based policy | an energy surface can hold several equally-good grasps where a regressor must average them |

Adding one: subclass `Policy`, decorate `@register("name")`, import it in
`handil/algos/__init__.py`. Diffusion Policy is the natural next addition —
it reuses ACT's chunked data path, so only the module is needed.

---

## Layout

```
handil/
  data/    schema · logs (schema-variance readers) · methods · actions · extract · dataset
  envs/    scene (MjSpec rebuild + gains) · rollout (control loop + success criteria)
  algos/   base (registry) · bc · act · ibc
  eval/    closed-loop evaluation
  train.py
scripts/   extract_data.py · train_policy.py · evaluate_policy.py · run_comparison.py
tests/
```

Run the tests with `uv run pytest -q`.

---

## Notes on the source logs

Things that will bite anyone reading `hand-control` logs directly:

- **8 distinct npz key-sets** across the archive, with `qpos` widths of 30 / 51
  / 58 (1-, 4- and 5-object scenes). Check `.files` and shapes per run.
- `norm_force` / `tan_force` are `(N,2)` on older runs and `(N,4)` on newer;
  finger order is `(thumb, index, middle, ring)`, so the missing middle/ring
  columns pad at the **end**.
- Join on **`t_wall`**, never `t` — sim time resets to 0 on Backspace and on
  MuJoCo `BADQACC` resets (real runs contain several).
- `pose_trace.npz` is the trajectory source, not the per-trial npz: the latter
  is a success-detection artifact and omits `q_robot` entirely in baseline modes.
- `max_slip_mm` is always 0.0 — `note_slip` is defined but never called.
