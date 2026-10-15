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
uv sync --extra dev --extra cuda          # torch (cu128), mujoco, pytest
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

## The hypothesis under test

**Does contact-aware collection produce better downstream policies than raw
teleop?** That is the question this repo exists to answer, and testing it needs
the retargeter held fixed — otherwise pipeline, retargeter and sample size vary
together and the result means nothing.

`anyteleop` is the only clean pair in the archive: `baseline+anyteleop` 63
episodes against `contact_aware+anyteleop` 76, same four objects, matched
solref-ablation spread. (This needs the full `logs/` tree with
`--include-ablations` — `ICRA_experiments` alone has just 3
`contact_aware+anyteleop` episodes.)

### Result: NOT supported on this data

> **Retraction.** Earlier versions of this file reported "contact-aware is 2-3x
> more learnable, no seed overlap". That was wrong, from two bugs of mine: a
> validation split over SAMPLES rather than episodes (which let windows ~10
> timesteps apart sit on both sides, so the metric measured interpolation within
> a trajectory), and only 3 seeds against an effect that needs 24.

Properly measured -- per-episode held-out split, BC-MLP, **24 seeds per arm**:

| | mean | sd | range |
|---|---|---|---|
| `baseline+anyteleop` | **0.0991** | 0.0465 | 0.030 – 0.178 |
| `contact_aware+anyteleop` | 0.1080 | 0.0790 | 0.009 – 0.298 |

effect **−0.0089** (baseline marginally better), Cohen d **−0.14**, Welch
t = −0.48, **p = 0.63**. A null result, with the sign *against* the hypothesis.

At 3 seeds the same comparison showed contact-aware ahead by d = 0.81. Bootstrapping
3-seed draws from these 24: **44% of draws make contact-aware look better by
chance**, and the +0.037 effect I originally reported sits at the 82nd percentile
of that distribution. There was no signal — only noise plus a leaky split.

BC-RNN under the per-episode split is likewise a tie (0.1434 vs 0.1408, d = 0.04).

#### What does survive

The *mechanism* is real and independent of the learning result, because it is a
direct property of the demonstrations rather than of any model: cross-episode
spread of hand posture at peak grasp falls 21–72% on three of four objects, and
object-in-palm spread falls 46–55%. Contact-aware collection genuinely produces
more repeatable grasps. It just does not make the data measurably easier to
regress at this sample size, and does not improve closed-loop behaviour.

#### What would be needed to test the hypothesis properly

With sd ≈ 0.047–0.079 between seeds and no detectable effect at n = 24, the
limit is now the **data**, not the seed count: 16 vs 23 episodes per cell is too
few for a difference this small to be resolvable. More
`baseline+anyteleop` / `contact_aware+anyteleop` episodes is the only route.

### Two caveats, stated plainly

**1. Part of the val-loss win is structural.** Contact-aware actions are
intrinsically smoother (action-step p95 0.0095 vs 0.0147; action std 0.276 vs
0.350), and smoother targets are easier to regress *regardless* of whether the
resulting policy is better at the task. Val loss is necessary evidence here, not
sufficient.

**2. The replay ceiling cannot adjudicate this, and the reason is our tooling.**
Policy-free replay gives `baseline` 12% pick / 4% success against
`contact_aware` 0% / 0% — which looks like the opposite conclusion. It is an
artifact:

- **51% of contact-aware rows have their finger targets replaced** by `qpos[t+1]`
  (vs 0% for baseline), because the pipeline's internal-force torques are not
  invertible from `qfrc_applied` — see *Actions are not logged* below.
- That collapses the commanded squeeze margin from **0.362 rad to 0.090 rad**, so
  the replayed grasp cannot generate force.
- The demos themselves generated *more* grip force than baseline (**126 N vs
  62 N**), so the data is not the problem.
- The contact-aware grasp *is* an internal-force controller, which the rollout
  env does not implement at all.

### Settled empirically: the phase one-hot buys nothing

`--phase-onehot` appends a 4-d phase one-hot (APPROACH/PICK/TRANSPORT/PLACE) as
an opt-in block, so the dataset is a strict superset. A/B over 3 seeds, 16
episodes/cell:

| observation | algo | baseline | contact_aware | ratio |
|---|---|---|---|---|
| 78-d (default) | BC-MLP | 0.00538 ± 0.00035 | 0.00265 ± 0.00005 | **2.03×** |
| 82-d (+phase) | BC-MLP | 0.00522 ± 0.00033 | 0.00269 ± 0.00009 | **1.94×** |
| 78-d (default) | BC-RNN | 0.00171 ± 0.00002 | 0.00167 ± 0.00025 | 1.02× |
| 82-d (+phase) | BC-RNN | 0.00166 ± 0.00006 | 0.00164 ± 0.00027 | 1.01× |

Every difference is inside the seed spread, and the contact-aware conclusion is
identical either way. So the default stays the two **stateless** dims
(`height_above_rest`, `hand_touching`), which the rollout env can reproduce
exactly — the one-hot cannot, since the real phase machine carries dwell timers
(`DWELL_S = 1.0 s`, `ARRIVAL_DWELL_S = 0.5 s`) that `observe()` has no state for.

Worth knowing how *partial* those two dims are, measured against the now-correct
labels: they carry **46%** of the phase label's information binarised (58% with
height finely binned), and the best achievable 3-class accuracy from them is
**78.5%** against a 39.7% majority baseline. The failure is specifically
PICK-vs-TRANSPORT — PICK includes *failed* attempts, where the object is lifted
then dropped, which is instantaneously identical to a TRANSPORT row with the
object momentarily airborne. That distinction is history, not state. The A/B
shows the policy does not need it.

### Learning the torque directly instead

The GRASP-phase position target is unrecoverable, but the **torque is logged
outright** — so rather than repairing a position label, learn the thing the
controller actually commanded. `--action-space torque` does this:

```
action = qfrc_applied[:23] - qfrc_bias[:23]      # N*m, no inversion, no gains
```

All 235 episodes extract with `action_source=logged_torque` and **zero GRASP
repair**, and the squeeze is now *in* the labels: contact-aware finger torques
have 2.4x the spread of baseline (0.683 vs 0.281 N*m std), exactly the
internal-force component the position label was throwing away.

**The result flips with architecture** (same controlled setup, 3 seeds):

| | baseline | contact_aware | winner |
|---|---|---|---|
| BC-MLP | 0.0413 ± 0.0018 | **0.0275** ± 0.0012 | contact_aware, 1.5x |
| BC-RNN | **0.0062** ± 0.0001 | 0.0096 ± 0.0020 | baseline, 1.5x |

Compare with position space, where contact-aware won on *both* architectures
(3.1x and 1.4x). So the contact-aware advantage is **not architecture-independent
once the squeeze torques are in the label** — a recurrent model extracts more
from the smoother baseline torque stream than from the contact-aware one. That is
a real finding about the data, not an artifact.

**Caveat on torque rollout.** Torque is an *open-loop* command: a small state
mismatch makes the commanded torque wrong for the actual state and the error
integrates twice, where a position command self-corrects. Open-loop torque replay
of the demonstrations diverges to 2-58 rad of arm error, so `action_space="torque"`
in `PickPlaceEnv` is correct for *executing a trained policy* (which re-reads
state every step) but useless for replaying a recorded stream.

### Internal-force squeeze in the env

`PickPlaceEnv(squeeze=True, gamma=...)` adds the contact-aware squeeze, so a
*position* policy can press rather than merely close. It **imports the source
repo's own `GraspController`** rather than reimplementing the null-space
allocation (`handil/envs/grasp.py`), taking only `internal_force_torques` +
`slip_correction_torques` and leaving the position hold to the env's P law —
exactly how `kinova_leap_pick_place.py` uses it in the teleop GRASP branch.

Two deviations worth knowing: `grasp_control/__init__` is bypassed (it imports
casadi via `constrained_ik`), and the contact frames come from **live**
fingertip/object geometry rather than the recommender's stored `rec_local`, which
is not in the logs. So it is a faithful internal-force controller, not a replay
of one specific recommended grasp.

**It works, and it is not sufficient.** Verified on contact-aware replay, gamma
sweep on `obj_009_gelatin_box`:

| gamma | peak force | max lift | touch frames |
|---:|---:|---:|---:|
| 0.5 | 2.9 N | 0.0001 m | 40 |
| 2.0 | 3.0 N | 0.0009 m | 83 |
| 5.0 | 2.4 N | 0.0069 m | 171 |
| 10.0 | 7.1 N | 0.0088 m | 183 |

Contact monotonically improves (40 → 183 frames, 4.6x) and so does lift, but it
plateaus just under the 1 cm pick threshold. With the robot's kinematics pinned to
the demonstration and gamma=20 the squeeze reaches **1210 N**, so the force
authority is certainly there — it over-squeezes and shoves the object out
(touch frames collapse to 4), against the demo's 168 N.

The binding constraint is **reach, not grip**: replay closes to 0.197 m of
hand-object distance where the demonstration reaches 0.145 m. Statically these
objects need only 0.23-3.6 N per finger, so a 5 cm positioning error cannot be
compensated by any amount of squeeze. Raising the ceiling further means improving
open-loop position tracking, not grip force.

### Position vs torque policies, with and without the squeeze

BC-RNN on `contact_aware+anyteleop` (76 episodes), closed-loop drift from the
demonstrated arm trajectory and closest hand-object approach:

| Policy action space | drift @50 | drift @200 | min \|obj_in_palm\| |
|---|---:|---:|---:|
| position | 2.26 rad | 4.19 rad | 0.443 m |
| **position + squeeze** | **1.12 rad** | **1.83 rad** | **0.303 m** |
| torque | 1.61 rad | 17.70 rad | 0.434 m |
| *(demonstration)* | — | — | *0.145 m* |

Two things fall out.

**The squeeze helps a trained policy, not just replay.** It halves drift at both
horizons and closes 14 cm of the reach gap. That is a better result than the
replay A/B suggested, and it is the main practical payoff of porting
`GraspController`.

**Torque is the wrong action space for BC here.** It trains fine (val 0.0070,
35 s on GPU) and stays numerically stable, but drifts 17.7 rad by step 200 —
4x worse than position. The reason is structural: a position command is
*self-correcting*, since a state error enters as a bounded restoring force, while
a torque command is *open-loop*, so an error integrates twice. Under BC's
covariate shift, torque amplifies the failure that position partly absorbs.

So the torque action space is the right **label** for the contact-aware GRASP
phase (it is the only faithful one — see above) but the wrong **command** for a
BC policy to emit. Those are separable choices, and a sensible next design is a
position policy with a learned squeeze magnitude, rather than full torque control.

### Three interventions, all 0% success — and what they ruled out

| intervention | drift@50 | drift@200 | success |
|---|---:|---:|---:|
| position BC (baseline) | 3.12 | 2.35 | 0% |
| position + DART (4×0.10) | 1.59 | 1.13 | 0% |
| **delta** BC | 1.20 | 4.44 | 0% |
| **delta + DART** (4×0.05) | **0.35** | **0.81** | 0% |

**DART noise injection** halves drift and then saturates (8 copies is worse than
4). **Delta actions** plus DART reduce drift@50 by **9×** versus position BC, and
the two compound. Neither produces task success.

**Start-pose jitter** (0 → 0.20 rad): 0% at every level for both plain BC and
DART, and nothing *degrades* with jitter, because there was no competence at
jitter=0 to lose. A DART policy's reach is flat across jitter where plain BC's
scatters, so its robustness did transfer — robustly to the wrong place.

#### The mechanism this found

With **absolute** joint targets, `corr(action, qpos) = 0.91` and
`|action − qpos|` is 0.016 rad median, so *"echo your current joint position"* is
a near-optimal regression solution. A BC-RNN trained that way predicts the
demonstrated action to **0.0018 rad teacher-forced** while commanding only
**13–39%** of the joint travel the task needs. It was never drifting off a good
trajectory — it had no trajectory. That is why neither recovery data nor start
variation could help.

Delta actions remove the shortcut: `corr(action, qpos)` falls **0.9125 → 0.0030**
and commanded travel rises to **0.65–2.62×** the demonstration. The
under-commitment is fixed; success is not.

#### Two measurement caveats, both ours

1. **A harness bug invalidated the first delta sweep.** `test_dart.py` built the
   env without `action_space`, so ~0.007 rad residuals executed as absolute
   targets. The tell was `reach` identical to 0.495 m across four different
   policies. Fixed, re-run, and pinned by a regression test.
2. **`min |obj_in_palm|` is not a hand-object clearance.** The palm *body*
   origin sits well behind the fingertips, so it carries a fixed offset:
   successful demonstrations themselves bottom out at a 0.126–0.161 m median.
   Measured by fingertip instead, plain position BC reaches **0.076 m** — closer
   than a demo replay's 0.185 m — so reach was never that policy's binding
   constraint. Do not read the `reach` column as a distance-to-grasp.

### ACT on delta actions: drift essentially solved, success still 0%

| space | chunk | DART | val | drift@50 | drift@200 | success |
|---|---:|---:|---:|---:|---:|---:|
| delta | 8 | 0 | 0.05038 | **0.25** | 1.26 | 0% |
| delta | 8 | 4 | 0.03204 | 0.34 | **0.82** | 0% |
| delta | 16 | 4 | 0.02842 | 0.25 | 1.06 | 0% |
| delta | 32 | 4 | 0.02907 | 0.26 | 3.06 | 0% |
| *position BC (reference)* | 1 | 0 | 0.00171 | *3.12* | *2.35* | 0% |

**drift@50 falls 3.12 → 0.25 rad, a 12× improvement**, and chunk 8 + DART gives
the best long-horizon number. Success does not budge.

#### The decisive measurement

Drift is no longer the problem, so the obstruction is elsewhere. Measured on the
best ACT policy over 400 steps:

| | ACT delta k8+DART | demo replay |
|---|---|---|
| closest fingertip–object distance | **0.347 m** | 0.019–0.143 m |
| frames in contact | **0 / 400** | 297–887 |
| peak finger force | **0.00 N** | 20–1311 N |
| object displacement | **0.0000 m** | lifts 0.095–0.175 m |

The policy **never arrives at the object** — zero contacts, zero force, object
never moves. Demo replays in the same env reach 1.9–14.3 cm with hundreds of
contact frames, so the environment and the grasp criteria are not what is
blocking it.

This also refutes an intermediate hypothesis of mine. The finger labels really
are degraded — the squeeze margin is 0.002 rad median, and 46% of contact-aware
rows have GRASP-repaired finger targets (`qpos[t+1]`, i.e. "follow where the
finger already went", which can track a closing hand but never command one). But
adding the internal-force squeeze at rollout changed nothing (`max_h` exactly
0.0000 at gamma 0, 5 and 15), because the hand is never near the object for a
squeeze to act on. **The hand is not the binding constraint; arriving is.**

So the honest summary after six interventions: drift is fixed 12×, under-commitment
is fixed, the shortcut is removed, and the policy still does not traverse the
~40 cm from its start pose to the object. That is the single remaining question,
and it is not explained by coverage, parameterisation, chunking, recovery data,
start-pose variation, or grip authority.

### What remains untested

Task-level success for the contact-aware pipeline. Both candidate fixes are now
implemented and measured: the squeeze helps substantially but leaves reach short
of the demonstration (0.303 m vs 0.145 m), and torque control makes things worse.
The remaining gap is open-loop position tracking under covariate shift, which
points at varied initial conditions / DAgger rather than at the controller.

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

> **Earlier numbers here were wrong.** They were measured with a trace-frame
> object index passed into a scene-frame parameter, so 3 of 4 objects were
> replayed against the *wrong object*. Fixed (address by name); remeasured below.

On the 93-episode nominal-only v2 dataset: **16% pick / 11% success** overall,
and **50% pick / 50% success** on `baseline+anyteleop` (n=16). Palm position
tracks the demonstrations to **1.4–2.1 cm p95**.

The per-pipeline split tracks **GRASP-repair fraction**, not pipeline quality:

| method | GRASP repair | min \|obj_in_palm\| | max lift | success |
|---|---:|---:|---:|---:|
| `baseline+anyteleop` | 0% | 0.139 m | 0.109 m | **50%** |
| `baseline+vwj` | 0% | 0.142 m | 0.072 m | 15% |
| `contact_aware+anyteleop` | 47% | 0.270 m | 0.000 m | 0% |
| `contact_aware+vwj` | 59% | 0.178 m | 0.003 m | 0% |

Every cell needing repaired finger targets fails; every cell with 0% repair
succeeds. Replay therefore still **cannot** adjudicate the contact-aware
pipeline — its labels, not its data, are what replay cannot execute.

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

### Status: results, and why closed-loop is still 0%

The full 3 methods x 4 algorithms matrix runs in **3.9 minutes** of GPU
training. Validation loss, 60 epochs, 12 eval rollouts per cell:

| | `baseline+anyteleop` (16) | `baseline+vwj` (13) | `contact_aware+vwj` (9) |
|---|---:|---:|---:|
| BC-MLP | **0.00535** | 0.03599 | 0.01752 |
| BC-RNN | **0.00174** | 0.02008 | 0.03720 |
| ACT | **0.03392** | 0.07765 | 0.04238 |
| IBC | **0.00008** | 0.00048 | 0.00069 |

**Read down the columns, not across the rows.** IBC's objective is an InfoNCE
cross-entropy over sampled negatives, not an MSE, so its numbers are not
comparable to the regression losses.

**This matrix does NOT test the contact-aware hypothesis** — see *The
hypothesis under test* above for the controlled comparison. The only
contact-aware cell here is `contact_aware+vwj` at 9 episodes, so pipeline,
retargeter and sample size all vary together; `baseline+anyteleop` ranking first
says nothing about the pipeline. It is kept only as a per-architecture sanity
check that the harness runs every cell.

**Closed-loop success is 0% in every cell, and the cause is diagnosed.** It is
not underfitting: fed demonstration observations, BC-RNN reproduces the
demonstrated actions to 0.004 rad median / 0.016 rad p95. But rolled out, the
arm's state deviates from the demonstrated trajectory by **1.6 rad within 50
steps**.

That is BC covariate shift, and it comes from the data rather than the
simulator: **every episode of a cell starts from exactly the same home pose**
(zero spread across all 7 arm joints, verified over all 63 `baseline+anyteleop`
episodes including ablations) in a deterministic scene. The policy has seen one
path through state space and no recovery states, so its first deviation is
fatal. Ground-truth replay through the same env reaches 31% on that cell, so the
env is not the blocker.

Two experiments already narrow the fix:

| Tried | Result |
|---|---|
| **ACT action chunking** (chunk=32) | early drift **0.183 rad at step 50 vs BC-RNN's 1.595** — a ~6x improvement, confirming chunking is the right lever. Still 1.15 rad by step 290; success 0%. |
| **Pooling 16 -> 63 episodes** (`--include-ablations`) | val 0.00174 -> 0.00166, i.e. no change. Expected in hindsight: the ablation runs add trajectories from the *same* start state, so they add volume but not state-space coverage. |

So the missing ingredient is **varied initial conditions**, not more data and
not more simulator fidelity. The remaining levers, in order: initial-state
jitter in the demonstrations (or DAgger-style aggregation using the rollout env,
which now works), longer action chunks, and keeping `timeout`/`abandoned`
episodes for off-manifold coverage.

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

## GPU

**Install with `--extra cuda`.** Plain `uv sync` resolves `torch` from PyPI to a
**cu130** build, which needs an NVIDIA driver >= 580. On a 570.x driver
(CUDA 12.8) that build reports `cuda.is_available() == False`, and since
`TrainConfig.resolved_device()` falls back to CPU automatically, training runs
~57x slower with no error — only a `UserWarning` buried in the log.

The `cuda` extra pins the **cu128** index, which works with a 12.8 driver and
supports the RTX 5090's Blackwell architecture (`sm_120`) that older CUDA builds
do not.

Measured on this box (2x RTX 5090), ACT for 60 epochs on 16 episodes:

| Device | Time | Best val |
|---|---:|---:|
| CPU | 4557 s (76 min) | 0.04064 |
| cu128 GPU | **80 s (1.3 min)** | 0.04058 |

Check before a long run:

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

If that prints `False`, you are about to train on CPU.

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
           test_pipeline_hypothesis.py · test_pipeline_replay.py
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
