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

# 1. Build the episode dataset from the ICRA benchmark runs
uv run python scripts/extract_data.py \
    --logs $HANDCONTROL_ROOT/logs/ICRA_experiments --out data/icra --outcomes success

# 1b. Calibrate: replay the demos themselves to get the env's ceiling
uv run python scripts/replay_demos.py --data data/icra --limit 20

# 2. Train one policy on one collection method
uv run python scripts/train_policy.py --data data/icra \
    --algo bc_rnn --retargeter anyteleop --pipeline baseline \
    --epochs 100 --out runs/base_anyteleop_bcrnn

# 3. Score it closed-loop in the real MuJoCo scene
uv run python scripts/evaluate_policy.py --run runs/base_anyteleop_bcrnn --episodes 20

# 4. Or run the whole comparison matrix (methods x algorithms)
uv run python scripts/run_comparison.py --data data/icra --algos bc_mlp bc_rnn act --episodes 20
```

The source repo is **read-only** to this one: its `logs/`, `models/` and YCB
assets are referenced by path, never copied, so the scene stays single-sourced.

---

## Scope: the ICRA benchmark runs

The work targets `logs/ICRA_*`, which is the clean, counterbalanced benchmark:
the **RoboCasa counter scene** (counter top `z = 0.86`), four YCB objects under
**sequential spawn** — exactly one object on the counter at a time, the other
three parked below the floor — each presented at the same point `[-0.35, 0.35]`
so initial conditions are identical across objects and runs, in a cyclic Latin
order. 35 runs, 4 collection methods.

Everything about that scene was measured from the runs themselves rather than
read off the source's constants, and two of those constants would have been
wrong: the ICRA runs use the RoboCasa counter, **not** the `pick_place` table at
`z = 0.625`, and they spawn at `[-0.35, 0.35]`, **not** the source's
`SEQ_SPAWN_XY = [-0.20, 0.45]`. Building the wrong one puts the object 18 cm
from where every demonstration reaches, and nothing then works.

---

## The three hard parts

Most of the engineering here is in making the source data learnable at all.
All three problems below were found by validating against the real archive, and
all three are easy to get silently wrong.

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

### 2. The palm-frame object pose has to be recomputed

`obj_in_palm` is the most task-relevant signal the policy gets — where the
object sits relative to the hand — and the logged column is **all-NaN in 41 of
44 ICRA episodes**, because the source only fills it for the proximate object
while a trial is live.

It is fully recoverable: `xpos`/`xquat` (palm) and `obj_qpos` (object
freejoints) are both logged, so

```
p_palm_frame = R_palmᵀ (p_obj − p_palm)      R_rel = R_palmᵀ R_obj
```

Validated against the rows where the source *did* log it: agreement to
**1e-5 m over ~11k rows**, the residual being float32 storage precision.

Picking *which* object is the active one matters here. Stowed objects are
parked at `z = -5` with collision off, and nothing holds them there, so they
free-fall for the rest of the run — real traces reach `z = -360`. The active
object is the one that stays above the floor, tested by row count rather than
by a median (a median over a falling object is meaningless).

### 3. The method label is lossy

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

44 successful episodes from the 15 `ICRA_experiments` runs, with no episode
needing the `next_qpos` action fallback:

| Collection method | Episodes | Steps |
|---|---:|---:|
| `baseline+anyteleop` | 16 | 19,758 |
| `baseline+vwj` | 13 | 17,421 |
| `contact_aware+vwj` | 9 | 6,961 |
| `contact_aware+anyteleop` | 3 | 2,316 |
| `baseline+dexpilot` | 3 | 6,632 |

Pointing `--logs` at the whole `logs/` tree instead yields 235 episodes across
195 runs and 13 method cells, but those mix scenes, object counts and table
heights, so they are not directly comparable. `logs/ICRA_ablations` adds 20 more
runs of the same scene under a solref stiffness sweep (`@s0p4/6/8`).

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
`base_link` re-posed onto the counter and yawed 90°, hand self-collision off,
YCB objects attached and settled under gravity, the inactive three stowed), then
applies the same P-torque control with no torque clamping. The recovered arm
gains match the source exactly: `[509.3, 675.5, 254.3, 324.0, 65.9, 65.0, 3.1]`,
verified identical under MuJoCo 3.6 and 3.14.

Success uses the source's own criteria from `trial_logger.py`: a lift of 1 cm
held for 1 s to confirm a pick, then contact-based containment in the bin
(contact with `bowl_base`, walls excluded, 0.2 s flicker latch) while settled
and released for 0.5 s.

### Calibrate before you trust a policy score

`scripts/replay_demos.py` drives the env with the **demonstrated** actions and
scores the result. That is the ceiling for any learned policy here, and it
separates "the policy is weak" from "the env is wrong" — a distinction that is
otherwise very easy to misread.

Across all 44 ICRA episodes it currently reaches **20% pick / 14% success**,
and **38% pick / 31% success** on the best-represented method
(`baseline+anyteleop`, n=16). Palm position now tracks the demonstrations to
**1.4–2.1 cm p95**, and on several episodes the replay's closest hand-object
approach matches the log to within a millimetre.

It is still well below 100% for a structural reason: replay is open-loop, so a
40-second contact-rich manipulation has no feedback to correct residual
tracking error, and a demonstration whose grasp was marginal will not survive
it. **Read policy numbers against this ceiling, not against 100%.**

#### What has been ruled out

Things tried that did **not** move the ceiling, so future work needn't repeat
them:

| Tried | Result |
|---|---|
| Arm velocity injection (the source's `qvel[:7] = _qdot_arm`) | no change — and the Kd term it pairs with is identically zero anyway |
| Servoing the arm onto the logged *achieved* positions instead of recovered targets | no change (~0.30 rad → same) |
| Interpolating the action stream 4× onto a finer grid | tracking p95 0.146 → 0.136 rad, success unchanged within noise |
| Per-step substep counts from each episode's own timestamps | helps, and is kept — `replay_episode(t=...)` |

Also checked: replay success does **not** correlate with any demonstration
feature — successes and failures have statistically identical peak finger force,
closest approach, duration, and GRASP-repair fraction. The residual is replay
dynamics, not demo quality.

One data-quality note that fell out of this: `baseline+dexpilot` demos show a
median *peak* finger force of 2299 N (against 50 N for `baseline+anyteleop`),
which is not physical for a LEAP hand and indicates deep interpenetration in
those recordings. Worth knowing before treating that cell as comparable.

#### The bug that dominated this number

Arm joints 0/2/4/6 are **continuous** — they rotate freely. The source writes
`model.jnt_range[j] = [-pi, pi]` for them, but only to bound its IK *sampling*;
it is not a physical stop. Enforcing it as one clamped joint 2 by ~0.17 rad,
which alone is ~9 cm of palm error — more than enough to miss a 5 cm object,
and enough to take replay success from 30% to 5%.

The tell was a static-hold test: commanded an off-home posture, every joint
converged to 1e-11 **except joint 2**, which sat at exactly π. The
demonstrations routinely drive joint 2 to 3.7–4.1 rad and joint 6 to 3.2–3.6 rad
on every ICRA run. Fixing it moved palm error from 90 mm to 0.2 mm on that test.
`tests/test_env.py` pins all of this.

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

### Status: first results, and why closed-loop is still 0%

The full matrix runs. On the offline axis it already separates the cells
cleanly (60 epochs, 12 eval rollouts per cell):

| Collection method | Algorithm | Episodes | Val loss | Closed-loop success |
|---|---|---:|---:|---:|
| `baseline+anyteleop` | BC-RNN | 16 | **0.0017** | 0% |
| `baseline+anyteleop` | BC-MLP | 16 | 0.0053 | 0% |
| `contact_aware+vwj` | BC-MLP | 9 | 0.0175 | 0% |
| `baseline+vwj` | BC-RNN | 13 | 0.0201 | 0% |
| `baseline+vwj` | BC-MLP | 13 | 0.0360 | 0% |
| `contact_aware+vwj` | BC-RNN | 9 | 0.0372 | 0% |

Two things are readable there: temporal context helps on the
best-represented cell (BC-RNN beats BC-MLP 3×), and `baseline+anyteleop` data
is 7–20× more learnable than the other cells. That is the comparison this repo
exists to make, on the offline axis.

**Closed-loop success is 0% everywhere, and the cause is diagnosed.** It is not
underfitting: fed demonstration observations, the BC-RNN reproduces the
demonstrated actions to 0.004 rad median / 0.016 rad p95. But rolled out, the
arm's state deviates from the demonstrated trajectory by **1.06 rad within 50
steps**, growing to 1.8 rad.

That is textbook BC covariate shift, made severe by the data: **all 16
demonstrations start from the identical home pose** and the scene is
deterministic, so the policy has seen exactly one trajectory through state
space and no recovery states at all. The moment it deviates it is off-manifold
with nothing to imitate.

The fixes are the standard ones, and none of them is an env problem:

1. **More and more varied data** — pool the `ICRA_ablations` runs (20 more of
   the same scene), keep `timeout`/`abandoned` episodes as negative-to-neutral
   coverage, and add initial-state jitter so the demonstrations span a volume
   rather than a line.
2. **Action chunking** — ACT is implemented and registered; chunked prediction
   is the standard remedy for exactly this compounding, and it has not yet been
   run in the matrix.
3. **DAgger-style aggregation**, which needs the rollout env that now works.

So: the infrastructure is done and validated, the offline comparison is
already informative, and closed-loop success needs the data and algorithm work
above rather than more simulator fidelity.

---

## Layout

```
handil/
  data/    schema · logs (schema-variance readers) · methods · actions · extract · dataset
  envs/    scene (MjSpec rebuild + gains) · rollout (control loop + success criteria)
  algos/   base (registry) · bc · act · ibc
  eval/    evaluate (closed-loop) · replay (demo calibration)
  train.py
scripts/   extract_data.py · replay_demos.py · train_policy.py
           evaluate_policy.py · run_comparison.py
tests/
```

Run the tests with `uv run pytest -q`.

---

## Notes on the source logs

Things that will bite anyone reading `hand-control` logs directly:

- **8 distinct npz key-sets** across the archive, with `qpos` widths of 30 / 51
  / 58 (1-, 4- and 5-object scenes). Check `.files` and shapes per run.
- Runs span **two different scenes** (`pick_place` table at `z=0.625`,
  `robocasa` counter at `z=0.86`) and two spawn regimes. The directory name
  does not tell you which; the logged object z does.
- **Stowed objects free-fall.** Sequential spawn parks them at `z=-5` with
  collision off and nothing holds them there, so by mid-run they are at
  `z=-77` … `-360`. Any "is this object active" test must tolerate that.
- `norm_force` / `tan_force` are `(N,2)` on older runs and `(N,4)` on newer;
  finger order is `(thumb, index, middle, ring)`, so the missing middle/ring
  columns pad at the **end**.
- Join on **`t_wall`**, never `t` — sim time resets to 0 on Backspace and on
  MuJoCo `BADQACC` resets (real runs contain several).
- `pose_trace.npz` is the trajectory source, not the per-trial npz: the latter
  is a success-detection artifact and omits `q_robot` entirely in baseline modes.
- `max_slip_mm` is always 0.0 — `note_slip` is defined but never called.
