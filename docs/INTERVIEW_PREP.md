# Interview prep: shared-autonomy teleoperation → imitation learning

Written for: you, preparing to talk about this work. Private — not for the repo's
public face, though it lives here so it stays with the code.

`FULL_ACCOUNT.md` is the technical record. This document is about *talking*
about it: what to say when asked, what you need to know cold, what a sharp
interviewer will push on, and where the field is going.

---

## PART 1 — The 90-second version

Memorise the shape, not the words.

> I built the imitation-learning half of a shared-autonomy teleoperation
> project. The system lets an operator teleoperate a 23-DoF arm-hand robot, and
> a solver certifies that each grasp can physically resist the task forces
> before committing to it. That certification raises task success from 69% to
> 94%.
>
> The open question was whether better demonstrations also make better training
> data. The field mostly assumes yes — I counted four systems that assert it
> against three that actually test it. So I ran the matched comparison.
>
> The answer was no, and strongly: policies trained on the certified data were
> 3 to 10 times worse across three architectures. What makes it more than a
> negative result is that I measured the demonstrations independently, before
> training anything, using a data-quality metric from a NeurIPS paper — and it
> predicted the policy outcome in both sign and rough magnitude. Two
> independent measurements agreeing is what turns an anomaly into a finding.
>
> The mechanism is a tradeoff that paper had already named: certification buys
> state diversity, which doesn't predict policy success, and pays in action
> divergence, which does.

**If they want one sentence:** *shared autonomy traded demonstration
learnability for task success, and I could measure the exchange rate.*

### The three follow-ups you will always get

**"Why is that interesting if it failed?"**
Because the failure is informative and the field has not tested the assumption.
A negative result from a matched comparison is worth more than a positive one
from an unmatched study — and three of the four systems claiming this benefit
never trained a comparison policy at all.

**"How confident are you?"**
Effect sizes |d| = 8–15, p < 0.0001, robust across three architectures, two
dataset versions, episode-length matching, object-mix balancing, neighbour
count and seed. What I would *not* claim: that it generalises beyond this task
family, or that the causal mechanism is established. It is localised, not
proven.

**"What would change your mind?"**
Three things, all cheap: the deterministic-planner arm closing the gap would
confirm the mechanism; it *not* closing it while divergence stays high would
mean my mechanism is wrong; a second task family showing the opposite sign
would mean the effect is specific to this setup.

---

## PART 2 — Concepts you need cold

If you cannot explain these without notes, the deep-dive questions will go
badly. Each entry is the thing itself, then the number that makes it concrete.

### Behavioural cloning and why it fails

Supervised learning on (observation, action) pairs. Trivial to implement; the
hard part is that it is *not* the same as supervised learning, because the
policy's own errors change its future inputs.

**Ross & Bagnell (AISTATS 2010), Theorem 2.1:** `J(π̂) ≤ J(π*) + T²ε`, with ε
the supervised error measured under the *expert's* state distribution. The
quadratic is in the horizon. The bound is **tight** — they construct a matching
system — and Rajaraman et al. (NeurIPS 2020) prove an information-theoretic
lower bound of the same order that holds even for a deterministic expert and
even with active querying.

The intuition worth saying out loud: one mistake puts you in a state the expert
never visited, where you have no training signal, so you make a bigger mistake.

### Covariate shift, and the fix that isn't free

DAgger (Ross, Gordon & Bagnell, AISTATS 2011) runs the learner, has the expert
label the states it actually reaches, and aggregates. **Do not say it gives
O(εT) unconditionally** — it is `uTε_N`, where `u` bounds the expert's
cost-to-go gap, and the paper itself admits *"u could be O(T)"* in the worst
case. It also needs an interactive expert, which a certified-grasp solver is
not: you cannot query it at an arbitrary state.

### Action divergence vs state diversity — the distinction this project turns on

**Belkhale, Cui & Sadigh, "Data Quality in Imitation Learning," NeurIPS 2023.**
Know this paper properly; it is the backbone of the result.

- **Action divergence** — at states the policy cannot distinguish, do the
  demonstrations disagree about what to do? *Predicts* policy success. Lower is
  better.
- **Transition diversity** — noise in the *environment* for a given
  state-action. **Not** demonstrator noise. People misstate this constantly;
  getting it right signals you read the paper.
- Theorem 4.1 bounds distribution shift by a sum of per-timestep divergences
  weighted by `(H−t)` — time-to-go. Early divergence propagates longest.
- Their finding that matters most here: **state diversity is NOT correlated
  with policy success**, and coverage bought from a noisier demonstrator can
  cost more in divergence than it buys.

**Why it matters for the pitch:** more variety in the data is not automatically
good. The usable test is whether the variation is *observable* — if the policy's
input contains the variable that explains the difference, two demonstrations
doing different things are both right. If it doesn't, they are irreducible
disagreement.

### The horizon problem and the three modern answers

- **Implicit BC** (Florence et al., CoRL 2021) — energy-based. The real claim
  is about **discontinuity** and set-valued functions, not only multimodality:
  explicit regression interpolates through values it should jump over.
- **ACT** (Zhao et al., RSS 2023) — action chunking plus a CVAE. Chunking
  predicts k steps at once, cutting the effective horizon k-fold and attacking
  the T² term directly. The CVAE handles "a human solves this differently each
  time."
- **Diffusion Policy** (Chi et al., RSS 2023) — expressive enough to represent
  arbitrary multimodal action distributions; +46.9% across 12 tasks. Currently
  the default strong baseline.

Note ACT uses **absolute** action targets, so "absolute actions are broken" is
too strong. The defensible version is "absolute actions with a short horizon
and proprioception in the observation."

### Data quality beats data quantity

**robomimic (CoRL 2021)** is your empirical anchor. Verbatim: *"results are
lower for MH datasets compared to PH datasets, even though the MH datasets
contain 100 more demos (300 vs 200)."* More data, worse policy, because it came
from six operators instead of one. Proficiency split is near-monotone: BC-RNN
on Square 58 → 36 → 12 → 2.

Also **Lesson L4**: 49–88% relative performance drops from adding
seemingly-benign proprioception to the observation. This is the published
version of the shortcut this project rediscovered.

### The shortcut problem

When the action is an absolute joint target and the observation contains the
current joint position, a policy scores well by echoing its input. Measured
here: `corr(action, qpos) = 0.91`, and policies commanded only **13–39% of the
needed travel**.

**Cite Wen et al., "Fighting Copycat Agents" (NeurIPS 2020)** — a shortcut that
*lowers held-out loss while destroying closed-loop performance*. Do **not** cite
causal confusion (de Haan et al. 2019); that is observation-side spurious
features, a different thing. I got this wrong initially and it is the kind of
error an interviewer who knows the literature will catch.

### Shared autonomy taxonomy

Worth knowing because "shared autonomy" covers four mechanically different
things:

| Kind | Mechanism | Examples |
|---|---|---|
| Segment-level trading | one agent owns the robot at a time | HITL-TAMP, PATO, Sirius |
| Continuous blending | signals combine every timestep | residual copilots, BlenDAgger |
| DOF-partitioned | standing split across the embodiment | NestDex, this pipeline |
| Shaping / feedback | human keeps authority, input constrained | TAPs, FABCO |

---

## PART 3 — What was done, and what was observed

### The system

Kinova Gen3 (7 DoF) + LEAP hand (16 DoF) = 23 actuated joints, MuJoCo, RoboCasa
counter, four YCB objects. Operator teleoperates through a retargeter
(AnyTeleop or an in-house DexPilot variant). The contact-aware pipeline adds a
wrench-feasibility certificate: a grasp commits only when the contact set can
resist the task wrench, with the squeeze magnitude γ\* from the same solve.

**Critical design property:** the certificate is **absent at deployment**. The
learned policy runs alone. That is what lets this experiment separate "the data
is better" from "the architecture is better" — a confound that HITL-TAMP's own
Appendix L raises and that none of its 40 citers since 2025 has resolved.

### The measurements, in order

1. **Task success** — 94% vs 69%, p = 0.0068. Widens to 100% vs 56% under the
   stiffness ablation, replicating the FGT paper's Table II.
2. **Policy learnability** — 8-fold cross-validation over *episodes*, 41 matched
   episodes per arm, retargeter fixed.
3. **Demonstration quality** — action divergence computed on the raw demos,
   before training, with no policy involved.
4. **Per-block decomposition** — the same loss split into arm (7 joints) and
   hand (16).

### The results

| Algorithm | baseline | contact-aware | gap | d |
|---|---|---|---|---|
| BC-MLP | 0.0828 | 0.6893 | 8.3× | −15.18 |
| BC-RNN | 0.0724 | 0.7364 | 10.2× | −10.31 |
| ACT | 0.1747 | 0.5235 | 3.0× | −10.74 |

Demonstration quality, measured independently:

| Metric | ratio (CA ÷ baseline) |
|---|---|
| Action divergence, arm | 2.12× worse |
| Action divergence, hand | 2.91× worse |
| …(H−t) weighted | 2.99× worse |
| State diversity | 1.20× higher |

Arm/hand loss split: **17.2× vs 7.0×**. The damage is in the approach, not the
certified grasp.

### Why the arm/hand split is the strongest single number

The finger labels were partly reconstructed — the GRASP-phase PD target was
never logged, so ~46% of contact-aware finger rows came from `qpos[t+1]`. The
obvious objection is "your labels are bad."

The **arm** labels were never reconstructed. Recovery there is exact, validated
to 0.003 rad over 1500 steps. And the arm is where the gap is largest. That
closes the objection with a number rather than an argument.

---

## PART 3b — Implementation details

The level of detail a technical interviewer will drill into. If you built it,
you should be able to answer these without looking.

### Actions are not logged — they are recovered

The single most consequential implementation fact, and the one most likely to
be probed. `hand-control` logs state, not commands. The action label is
reconstructed by inverting the PD law:

```
target = qpos + (qfrc_applied − qfrc_bias) / kp
```

- `qfrc_applied − qfrc_bias` isolates the controller's torque from gravity and
  Coriolis terms.
- **`kp` is analytic, never fitted** — `diag(M) · ω²` with ω = 30 rad/s,
  giving `[509.3, 675.5, 254.3, 324.0, 65.9, 65.0, 3.1]` for the arm. An
  earlier version fitted the gains and was wrong by a factor of 1000.
- Validated by replaying the recovered actions through the simulator:
  **0.003 rad drift over 1500 steps**.

**The exception, and why it matters.** During the contact-aware GRASP phase the
commanded torque is `kp_eff · (q_grasp_hold − qpos) + τ_internal + τ_slip`, and
neither `q_grasp_hold` nor the time-varying `kp_eff` was logged. Inverting as
if it were pure PD gives finger targets up to 20 rad. So those rows are
repaired with `qpos[t+1]` ("follow where the finger actually went"), affecting
~46% of contact-aware rows — which is exactly the objection the arm/hand split
answers, since the arm is never repaired.

`action_source` is recorded per episode (`logged_grasp_hold` vs
`inverted_torque` vs `next_qpos`) so any analysis can filter on it.

### Observation space: 78 dimensions

| Field | Dims | Note |
|---|---|---|
| `q_arm`, `q_hand` | 7, 16 | joint positions |
| `qd_arm`, `qd_hand` | 7, 16 | joint velocities |
| `obj_in_palm` | 3 | object position **in the palm frame** |
| `obj_quat_in_palm` | 4 | |
| `palm_pos`, `palm_quat` | 3, 4 | world pose |
| `norm_force`, `tan_force` | 4, 4 | per-fingertip normal and tangential |
| `obj_id_onehot` | 4 | keyed on the object **name** |
| `obj_size` | 3 | mesh-AABB half-extents |
| `obj_mass` | 1 | |
| `height_above_rest` | 1 | stateless task-stage proxy |
| `hand_touching` | 1 | binary |

Four design decisions behind that table, each with a reason an interviewer can
challenge:

- **`obj_in_palm` is recomputed, not read.** The trace's own column is filled
  only for the proximate object during an active trial — all-NaN in 41 of 44
  ICRA episodes. It is reconstructed from logged world poses as
  `R_palmᵀ (p_obj − p_palm)`.
- **`obj_size` uses mesh-AABB half-extents, not `geom_size`.** For the mug,
  `geom_size` is hull 1 of a 43-hull convex decomposition — a 1.2 cm sliver
  that would tell the policy the mug is the *smallest* object.
- **Object identity is in the observation** because a matched-neighbour
  analysis found only **32%** of baseline nearest neighbours shared the query's
  object. Without it the policy cannot tell which object it is reaching for.
- **`phase_onehot` was deliberately dropped.** PLACE is exactly 1 row/episode
  and 81.5% of those have `|obj_in_palm| > 1 m` (the trial has already torn
  down); PICK is absent from 115 of 222 episodes because dexpilot-lineage modes
  advance silently, so its presence encodes the *teleop mode* rather than
  physics. It is available behind `--phase-onehot` for anyone who wants it.

### Architectures and their hyperparameters

| | Config | Loss |
|---|---|---|
| BC-MLP | 512×512 MLP, 1 obs → 1 action | MSE |
| BC-RNN | 2-layer GRU, hidden 512, **10-frame** obs window, `chunk=1` | MSE |
| ACT | d_model 256, 8 heads, 4 enc + 4 dec layers, **chunk=16**, latent 32, KL weight 10 | L1 + β·KL |
| IBC | 512×512×512 energy net, 256 negatives, 3 refine iters | InfoNCE |

**Be precise about BC-RNN**: it has a 10-frame observation *history* but
`chunk=1`, so it predicts one action. History is not chunking. The difference
from ACT is three-fold — sequence model vs transformer, 1 action vs 16, and
deterministic MSE vs a CVAE latent.

**Losses are not comparable across algorithms.** MSE for the BC pair, L1+KL for
ACT. Only the within-algorithm ratio means anything, which is why every result
is quoted as a ratio.

### Training protocol

- AdamW, lr 1e-4, weight decay 1e-4, cosine schedule with `T_max = epochs`
- 60 epochs, batch 256, gradient clipping at norm 1.0
- **Best validation loss across epochs**, not final — effectively early
  stopping with a free validation set, applied identically to both arms
- 8-fold CV over episodes × 6–12 seeds per condition

Actions are z-scored per dimension with **one normaliser fit over both
conditions pooled**. Per-condition normalisation would rescale each arm by its
own spread and destroy the comparison — a condition with more variable actions
would be divided by a larger number and look artificially good.

Scale, for converting results back to physical units: 1 normalised unit ≈
0.5 rad at median joint scale (arm std 0.28–0.89 rad, hand 0.12–0.71). So
baseline's 0.0828 MSE is RMSE 0.29 ≈ 0.14 rad ≈ 8° per joint; contact-aware's
0.689 is RMSE 0.83 ≈ 0.42 rad ≈ 24°. **An RMSE near 1.0 in z-scored units means
the policy is barely beating the dataset mean** — at 0.69, it extracts only
~31% of the variance from contact-aware data against ~92% from baseline.

### The action-divergence metric, precisely

```
for each of 4000 sampled query frames:
    find k=8 nearest neighbours in NORMALISED observation space,
        restricted to OTHER episodes
    divergence = mean |a_neighbour − a_query|, split arm / hand
```

Three details worth knowing:

- **Cross-episode is the critical restriction.** Within an episode the nearest
  neighbours of a frame are its temporal neighbours, whose actions differ by
  one control step (~0.006 rad). That reports `dt`, not disagreement.
- **Two different spaces.** Neighbour *search* is in z-scored observation
  space (so newtons, metres and radians are comparable); the divergence *value*
  is in raw radians, so it reads physically — baseline demos disagree by
  ~0.20 rad (12°) per arm joint, contact-aware by ~0.43 rad (25°).
- **The `(H−t)` weighting** implements Belkhale's Theorem 4.1: divergence early
  in an episode is weighted by time-to-go, because it propagates further.

Robustness checks run: stable across seeds, insensitive to `k` over {4, 8, 16}
(1.74–1.97× on v3), and with the object mix equalised at 6 per object per arm,
1.98× ± 0.14 with all 8 bootstrap draws above 1.

### Data pipeline

- One episode = one trial, sliced from `pose_trace.npz` by the trial's
  **wall-clock** window. Not sim time: that resets to 0 on Backspace and on
  MuJoCo BADQACC auto-resets, and real runs contain up to 4 such resets.
- `SCHEMA_VERSION` in the manifest; the loader refuses mismatched shards,
  because the `.npz` shards are bare float arrays with no self-describing
  header and a stale one would silently mis-shape every batch.
- Collection method is resolved from **two** sources: pipeline from the event
  (authoritative), retargeter from the directory name (the event drops it).
  Disagreements are surfaced as `method_conflict`, never silently resolved.
- `collapse_still` removes rows where max |dq| over 23 joints is below 1e-4
  rad, keeping the first frame of each pause. Selectivity: **94.6% of PLAN,
  13.8% of REACH, 1.0% of GRASP** — it targets the planner's dead-time hold
  almost exactly without naming a phase, and leaves a continuous trajectory so
  chunked and recurrent policies can still use it.

### Environment and evaluation

`PickPlaceEnv` reconstructs the RoboCasa scene with MjSpec surgery. Two fixes
worth being able to explain:

- **Continuous joints.** Kinova joints 0/2/4/6 are continuous, but the model
  declares finite ranges. Clamping them to ±π broke replay; setting
  `jnt_limited = 0` took replay success from 5% to 30%.
- **The object-index frame bug.** `EpisodeMeta.obj_idx` indexes the log's spawn
  order; `set_active_object()` indexes `scene.object_bodies`. They differ for
  3 of 4 objects, so most replays ran against the wrong object. Fixed by
  addressing objects by **name**; best cell went 31% → 50%.

Rollouts run ≥ 2500 steps: at 400 every policy scores 0%, the first real pick
appears at 2500, and 4000/6000 are identical.

### Scale of the thing

~5,500 lines across `handil/` and `scripts/`, 93 tests, 36 commits. Four
algorithms, four evaluation scripts, two deck generators. Roughly 151 episodes
extracted from 213 run directories.

---

## PART 4 — Likely questions, and answers

### On the result

**"Isn't this just a bug in your pipeline?"**
It might have been, and four earlier results in this project *were* bugs —
which is why I measure both ends of the chain now. The specific reassurance: the
demonstration metric and the policy metric are computed by different code paths
over different representations, and they agree in sign and roughly in
magnitude. A bug producing that coincidence would have to be unusually
cooperative.

**"Your closed-loop success is 0% for both conditions. Why believe anything?"**
Because 0 vs 0 is an uninformative tie, not evidence of equivalence. The offline
comparison is the one that discriminates, at |d| = 8–15. I would not claim it
predicts deployment performance — I would claim it predicts which data is
easier to fit, which is the decision in front of me when choosing what to
collect next.

**"Isn't the metric circular — you chose one that captures what you found?"**
The ordering answers this. The metric and the *direction* of its prediction come
from a 2023 paper, fixed before I computed anything. I then checked whether it
matched the policy result. A metric chosen to fit would not also have had to
survive that test.

**"41 episodes is small."**
Agreed, which is why the metric choice matters. K-fold over episodes is 12×
more stable than a single split (seed sd 71% → 8% of the mean), and the effect
sizes are far outside noise. Whether it *generalises* is open and I say so.

### On the methodology

**"Why validation loss rather than task success?"**
Because success is 0% for both, so it has no discriminative power. Validation
loss on held-out *episodes* is the sharpest available signal, and the sign is
what matters. I would rather report a discriminative proxy honestly than a
non-discriminative headline.

**"Why episode-level splits?"**
Windows ten timesteps apart in one episode are near-duplicates. A per-sample
split put them on both sides of the train/val boundary and made losses look
25–85× better than they were. I found that by not believing a good number.

**"How do you know 8 folds is enough?"**
Measured: single split at 16 episodes gave seed sd of 0.0907 (71% of the mean);
8-fold gave 0.0074 (8%). The fold-level sd stays high — episodes genuinely
differ — but CV averages that rather than sampling it.

### On the mechanism

**"What actually causes it?"**
Honestly: localised, not established. Ruled out — reconstructed finger labels
(arm-only training made it *worse*, 3.19× → 8.82×), sample size (16 → 63
episodes moved loss 0.00174 → 0.00166), grip force (1210 N available against
3.6 N needed), and the planner's dead-time hold (removing it recovers only
~15%). What remains is that the post-hand-over segment is where demonstrations
disagree, and I have not isolated why within it.

**"Isn't more diversity good for learning?"**
This is the best question and you should welcome it. Two things get called
variance: state diversity (how much of the space you visit — good, and
uncorrelated with success in Belkhale's measurements) and action divergence (at
the *same* observation, do demos disagree — always bad). Contact-aware bought
1.2× of the first and paid ~3× in the second. The usable test is observability:
object position is in the 78-D observation, so reaching differently for
differently-placed objects is informative variation. Variation the policy
cannot see in its input is pure divergence.

### On your own errors — do not hide these

An interviewer who senses you are polishing will dig. Volunteering a couple of
these is stronger than being caught:

- **A label-resolution bug hid 25 of 56 episodes.** `run_config.mode` logs a
  *normalised* mode that drops the retargeter, and I trusted it over the
  directory name. Found it by checking an unrelated question. Fixed, regression
  test added, everything re-run — and the effect got *larger*.
- **I misdiagnosed RRT as "slow."** It wasn't; the straight-line path check had
  started failing, so the planner now actually ran. The code comment literally
  warned *"a fast plan now means the line was clear, not that RRT was quick"* —
  and I fell for it before reading carefully.
- **I compared two different events** when claiming baseline had more hand-over
  variance. Baseline has no hand-over at all; my script silently fell back to a
  success event.
- **I cited the wrong paper** for the action-space shortcut (causal confusion
  rather than copycat agents).

Frame these as the reason the final result is trustworthy, not as apology.

---

## PART 5 — Design choices worth defending

| Choice | Why | What it costs |
|---|---|---|
| K-fold CV over **episodes** | A single split holds out 1–2 episodes, so the loss reports *which*. Seed sd 65–154% not shrinking with data. | 8× the compute |
| **Matched** episode counts + fixed retargeter | Otherwise pipeline, retargeter and sample size vary at once | Discards surplus data |
| Pooled normaliser across conditions | Per-condition would rescale each arm by its own spread and destroy the comparison | — |
| **Cross-episode** neighbours only | Within an episode nearest neighbours are consecutive frames — measures dt, not disagreement | Fewer usable queries |
| Best-epoch rather than final loss | Effectively early stopping with a free val set; applied identically to both arms | Mildly optimistic absolute values |
| `collapse_still` over dropping a phase | Leaves a continuous trajectory, so chunked/recurrent policies can use it | Slightly more complex |
| Measure data *and* policy | Either alone is weak; agreement between them is the evidence | More work |

**The meta-choice worth stating:** I built the measurement before trusting the
result, and rebuilt it twice when it turned out to be measuring the wrong thing.
That is the habit worth demonstrating in an applied-research interview.

---

## PART 6 — State of the art, as of late 2026

Be able to sketch this; it is what separates "ran experiments" from "knows the
field."

### Policy architectures

The field has consolidated on **expressive, chunked, multimodal** policies.
Diffusion Policy and ACT are the defaults; flow-matching variants are the
current speed play. The open problems are *inference latency* on real hardware
and *long-horizon* composition, not raw expressivity.

### Vision-language-action models

The dominant direction: pretrain on heterogeneous robot data plus web-scale
vision-language, then adapt. **RT-X / Open X-Embodiment** established that
cross-embodiment pooling helps. **π0** and successors brought flow-matching
VLAs. The live debate is whether scaling data fixes generalisation or whether
the data is too narrow in the ways that matter.

Relevant to your work: **DATAFARM (2026)** found raw planner-generated
trajectories give *"surprisingly little benefit"* for VLA fine-tuning — 8.3% vs
61.7% for human teleop — attributed to behavioural distribution mismatch. That
is your result, independently, in the VLA setting.

### Data

This is where the field is most active and least settled, and where your work
sits:

- **Collection at scale** — DROID, Open X-Embodiment, fleet learning. The
  bottleneck is human time, which is what drives interest in shared autonomy.
- **Curation** — DemInf (RSS 2025) uses mutual-information estimation, 5–10%
  improvement on robomimic. But the curation-metric literature is **actively
  contested**: a 2026 audit found spectral arc length may be an episode-length
  proxy (AUROC 0.952 vs 0.957 for length alone), and that detection accuracy
  and policy benefit can *decouple* — the best-detecting metric gave the worst
  policy. Those audits are single-author preprints, so weight them accordingly.
- **Simulation and generation** — MimicGen-style augmentation, real-to-sim-to-real.

**The honest gap:** there is still no reliable *a priori* metric that predicts
downstream policy success. Belkhale gives the formalism and concedes their
single-timestep action variance misses long-horizon multimodality. That gap is
exactly what your work speaks to.

### Shared autonomy for data collection

The specific area you can claim expertise in. The counting, which you did
yourself:

- **At least four systems assert** that assisted collection improves
  demonstration data without training a comparison policy (PATO, NestDex,
  DexGrasp Copilot, others)
- **Exactly three matched comparisons exist**, all 2026 — and the broadest
  (HATS, 7 tasks) finds *parity*
- **HITL-TAMP's Appendix L** shows its gain is largely attributable to
  test-time gating rather than data quality, and **none of its 40 citers since
  2025 has revisited it**
- Two 2026 papers (BRIDGE, DATAFARM) find assistance-provenance effects that
  actively harm learning — and **neither cites the shared-autonomy
  data-collection line at all**

That last point is the one to lead with: the two literatures have not met.

---

## PART 7 — Where robotics is going (your take)

Have a view. Hedging reads as not having thought about it. These are defensible
positions, with the reasoning attached so you can argue rather than assert.

**1. Data quality becomes a first-class research problem, not a preprocessing
step.** The field spent 2023–2025 scaling collection and is now finding that
more data is not monotonically better — robomimic showed it in 2021, and
large-scale efforts keep rediscovering it. The missing piece is a metric that
predicts downstream success *before* you train. That is a concrete, tractable
problem and it is where I would put effort.

**2. The human in human-robot interaction is the bottleneck, in both
directions.** Collection is gated by operator time, which drives shared
autonomy; but shared autonomy changes the data distribution in ways nobody is
measuring. My result is one instance. The general principle — *assistance that
helps the operator does not automatically help the learner* — probably
generalises, and the field has not internalised it.

**3. Evaluation is the quiet crisis.** Offline metrics do not predict
closed-loop success; closed-loop evaluation is expensive and
high-variance; sim-to-real gaps make simulation results contestable. I hit this
directly — 0% closed-loop for both conditions meant my only discriminating
signal was offline. Better evaluation protocols would unblock more progress
than better architectures.

**4. Dexterity remains genuinely unsolved, and contact is why.** Parallel jaws
work because the contact model is simple. Multi-finger hands fail for reasons
that are physical, not algorithmic — friction cone estimation, contact
transitions, tactile sensing that is still not good enough. The certification
approach I worked on is one attack; learned tactile representations are
another.

**5. A prediction worth stating:** the next significant gains come from
*closing the loop between data collection and policy learning* — systems that
measure what the learner actually needs and steer collection toward it, rather
than collecting first and filtering after. That is active learning applied to
demonstration collection, and the pieces (uncertainty-gated collection,
data-quality metrics, assisted teleoperation) exist separately but have not been
assembled.

### If asked "what would you work on?"

The honest answer, which is also the strong one: **the gap between what a
demonstration looks like to a human and what it looks like to a learner.** My
project is one data point — a pipeline that demonstrably helps the operator and
demonstrably hurts the policy. Generalising that into a predictive metric, and
then into a collection system that optimises for it, is a research programme
rather than a paper.

---

## PART 8 — Framing for an HRI / applied-scientist role

What this project demonstrates, in the terms those roles are hiring for:

- **Experimental design under confounds** — matched counts, fixed retargeter,
  pooled normalisation, episode-level splits. Each chosen because a weaker
  version gave a wrong answer.
- **Measurement before belief** — rebuilt the evaluation metric twice on
  finding it measured the wrong thing; retracted published-internal claims when
  the data refused them.
- **Human-in-the-loop systems end to end** — teleoperation, retargeting,
  shared control, the data it produces, and what a learner does with it.
- **Literature fluency** — situated the result against ~25 papers, found the
  one that predicted the failure, and identified that the counter-evidence sits
  in a literature the main one does not cite.
- **Negative results handled well** — the finding is a falsified hypothesis, and
  the write-up is stronger for saying so plainly.

**The sentence to have ready:** *"I'm the person who checks whether the thing
everyone assumes is actually true, and I'd rather find out it isn't."*
