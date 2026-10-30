# Learning from shared-autonomy teleoperation data: the full account

Written for: you, as the record of what was tried and what it means. Also the
source document for the talk.

Last updated 2026-10-23. Numbers marked ⏳ are still running.

---

## 0. The question and the chain

The FGT pipeline certifies that a grasp can resist the task wrench before
committing to it. It demonstrably improves the *task*: **94% vs 69%** success
(p = 0.0068), widening to 100% vs 56% under the object-stiffness ablation,
which independently replicates the FGT paper's Table II on data collected here.

The hypothesis under test was a chain:

> certified grasps → better demonstrations → better policies

- **Link 2 → 3** (better demos → better policies) is among the best-supported
  claims in imitation learning. Taken as given.
- **Link 1 → 2** (certification → better demos) is asserted widely and tested
  rarely. This is what the project measured.

Both links are measurable separately, so a failure can be localised rather
than blamed on the learner.

---

## 1. What prior literature predicted

### Why consistency should matter at all

**Ross & Bagnell (AISTATS 2010), Thm 2.1:** `J(π̂) ≤ J(π*) + T²ε`, where ε is
supervised error under the expert's own state distribution. The bound is
**tight**, and Rajaraman et al. (NeurIPS 2020) prove a matching
information-theoretic lower bound that holds even for a deterministic expert.
So horizon-quadratic compounding is fundamental to offline BC, not an artifact
of loose analysis.

**Belkhale, Cui & Sadigh (NeurIPS 2023), Thm 4.1** adds *where* it hurts: the
distribution-shift bound carries an `(H−t)` time-to-go weight. Divergence early
in an episode propagates through everything after it.

**robomimic (CoRL 2021)** makes it empirical and is the single most relevant
prior result: *"results are lower for MH datasets compared to PH datasets, even
though the MH datasets contain 100 more demos (300 vs. 200). This most likely
stems from the presence of suboptimal and multimodal data."* More data, worse
policy, because it came from six operators instead of one. Their proficiency
split is near-monotone: BC-RNN on Square falls 58 → 36 → 12 → 2.

### What the shared-autonomy literature predicted

Three matched assisted-vs-unassisted downstream comparisons exist, all 2026:

| Work | Result |
|---|---|
| Sha et al. (arXiv:2603.17016) | 18/20 vs 7/20 grasp; 11/20 vs 1/20 insertion |
| TAPs (arXiv:2512.04960) | Square 0.372 → 0.531; real unscrewing 38% → 67% |
| HATS (arXiv:2606.16491) | **parity**: 71.3 vs 70.3, 7 tasks |

Against at least four systems (PATO, NestDex, DexGrasp Copilot, others) that
*assert* the data benefit without ever training a comparison policy. AnyTeleop
(RSS 2023) — the retargeter used here — already argues the smoothness version:
*"the state-action pairs are more consistent and easier to be consumed by the
network."*

**So the literature's expectation was that contact-aware data should train
better policies.**

### The one paper that predicted the opposite

Belkhale et al. identify two properties: **action divergence** (demos
disagreeing at states the policy cannot distinguish — predicts success) and
**transition diversity** (noise in the *environment*, not the demonstrator).
They report **state diversity is uncorrelated with policy success**, and
identify a tradeoff where coverage bought from a noisier demonstrator costs
more than it buys. Their words on prior work: it *"aim[s] to maximize state
diversity through shared control"* while leaving the action side unexamined.

That sentence describes this pipeline, written three years before the
experiment. It was the one clear prediction that the hypothesis would fail.

---

## 2. What was measured, in order

### 2.1 First attempt: BC on v2, absolute action targets

**Expected:** contact-aware data trains better policies.
**Got:** 0% closed-loop success for every policy, both conditions.

Offline losses looked fine, so the failure was not fitting. Diagnosis found
`corr(action, qpos) = 0.91` — with absolute joint targets the policy can score
well by echoing its own current position, and measured policies commanded only
**13–39% of the needed travel**.

**Fix:** delta (residual) actions remove the shortcut.
**Status:** this turned out to be a **rediscovery**, not a finding. robomimic's
Lesson L4 reports 49–88% relative drops from adding proprioception to the
observation; "Demystifying Action Space Design" (ICML 2026) establishes
chunk-wise delta beats absolute over 13,000 rollouts. The right citation for
the mechanism is **Wen et al., "Fighting Copycat Agents" (NeurIPS 2020)** — a
shortcut that lowers held-out loss while destroying closed-loop performance.
(I originally cited causal confusion, de Haan et al. 2019. That was wrong:
causal confusion is observation-side, not action-parameterisation.)

### 2.2 The metric was broken twice before it was trustworthy

**Per-sample validation splits.** Windows ten timesteps apart in one episode
are near-duplicates; splitting over samples put them on both sides and made
losses look **25–85× better** than they were. Fixed with `episode_split`.

**A single held-out episode.** With `val_frac=0.1` the validation set is 1–2
episodes, so the loss largely reports *which* episode that was — measured
spread of 24× at 23 episodes, and seed sd of **65–154% of the mean that did NOT
shrink with more data**. Fixed with `episode_folds`: 8-fold CV over episodes
made the metric **12× more stable** (seed sd 71% → 8% of the mean).

**Consequence:** I had earlier advised collecting 40–50 episodes per cell
because "more episodes shrink noise." That was falsified by measurement — the
noise was a metric artifact, not a sample-size problem.

**Two other bugs invalidated whole result sets:** a missing `action_space`
argument silently invalidated an entire delta sweep (tell: `reach` identical to
0.495 across four policies), and an object-index frame mismatch meant 3 of 4
objects were replayed against the wrong object.

### 2.3 The headline comparison (v3, 31 matched episodes/arm)

| Algorithm | baseline | contact-aware | gap |
|---|---|---|---|
| BC-MLP | 0.0968 | 0.3477 | 3.6× |
| BC-RNN | 0.0851 | 0.3362 | 3.9× |
| ACT | 0.1896 | 0.3199 | 1.7× |

**Deviation from expectation:** the literature predicted contact-aware would
win. It lost, across three architectures, with |d| = 8–10 and p < 0.0001.

Notable: BC-RNN is the architecture robomimic identifies as *most tolerant* of
mixed-quality data. It did not rescue this. ACT — chunking plus a CVAE latent,
the design intended for exactly this failure mode — narrowed the gap most and
still lost.

### 2.4 Link 1 → 2 measured directly

Rather than infer data quality from policy performance, measure it on the raw
demonstrations with Belkhale's own metric: action disagreement among nearest
**cross-episode** observation neighbours. (Cross-episode because within an
episode the nearest neighbours are consecutive frames, which would report the
control timestep, not disagreement.)

| Metric (lower better) | baseline | contact-aware | ratio |
|---|---|---|---|
| Action divergence, arm | 0.208 | 0.362 | 1.74× worse |
| Action divergence, hand | 0.108 | 0.200 | 1.86× worse |
| …(H−t) weighted | 0.278 | 0.584 | 2.10× worse |
| Action variance, arm | 0.0076 | 0.0137 | 1.80× worse |
| Action variance, hand | 0.0053 | 0.0042 | 0.80× **better** |
| State diversity | 5.38 | 6.53 | 1.22× higher |

**Link 1 → 2 fails**, and it fails in exactly the way Belkhale predicted: +22%
state diversity (uncorrelated with success) bought at ~2× action divergence
(predicts success). The pipeline optimised the metric that doesn't matter and
degraded the one that does.

This is the most important methodological point in the project: **the data
metric predicted the policy result, in sign and roughly in magnitude, before
any policy was trained.** Either measurement alone would be weak; the agreement
between two independent ones is what makes it a finding.

---

## 3. Intermediate hypotheses, and how each died

### H1: "The repaired finger labels cause the gap"

The contact-aware GRASP phase commands `kp_eff * (q_grasp_hold - qpos)`, and
neither `q_grasp_hold` nor `kp_eff` was logged, so finger targets had to be
reconstructed from `qpos[t+1]` for ~46% of contact-aware rows. Obvious suspect.

**Test:** train on the arm only (7 joints, labels never reconstructed).
**Result: the gap got WORSE — 3.19× → 8.82×.** Refuted, by my own hypothesis's
own test.

That also pointed the way: the damage is in the arm, not the fingers.

### H2: "Not enough data"

**Test:** pool 16 → 63 episodes.
**Result:** val loss 0.00174 → 0.00166. Nothing. Refuted.

### H3: "Grip force too weak"

**Test:** measure squeeze against demand.
**Result:** squeeze already reaches 1210 N against the demos' 168 N, and these
objects need 0.23–3.6 N to hold statically. Reach binds, not grip. Refuted.

### H4: "Hand-over timing variance is the mechanism"

Lock-in occurs at a median 76% through the trial with a p10–p90 spread of 28
points, so the certificate takes over at a different stage every episode. A
policy conditioned on observation alone cannot infer how far through the
hand-over it is — state aliasing.

**Partly refuted / partly contaminated.** Two problems surfaced:

1. **The baseline comparison was invalid.** Baseline has no `rec_accept` event,
   so my script silently fell back to `pick_confirmed` — a *success outcome*,
   not a hand-over decision. Comparing 56.5% to 11.8% compared two different
   events. The contact-aware 28-point spread stands on its own; the comparison
   does not. Correct framing: **baseline has no hand-over at all**; this is a
   failure mode that only exists in contact-aware.
2. It was superseded by H6 below.

### H5: "RRT's random seed injects unobservable variance"

The contact-aware pipeline plans with RRT after lock-in and replays the
waypoints kinematically, and `extract.py` never filters those rows — so
**59% of contact-aware frames are planner output** (range 24–86%), and 0% of
baseline frames.

**Retracted as stated.** The REACH segment *is* where demos disagree most
(cross-episode palm spread 14–21 cm, end-points scattering 20–33 cm), but RRT's
start AND goal both differ between episodes — the operator hands over from a
different pose, and the certificate picks a different grasp. So this does not
isolate planner randomness from goal variation. Measured: the post-hand-over
segment is the divergent one. Not measured: why, within it.

### H6: "The PLAN hold is dead time the policy cannot model"

Found by breaking episode length down by phase:

| contact-aware group | median steps | PLAN | REACH | GRASP |
|---|---|---|---|---|
| September runs | 810 | **2** | 537 | 503 |
| October runs | 2222 | **851** | 1138 | 419 |

PLAN is the arm holding still while RRT solves. It went from 2 steps to ~850 —
a variable-length stall whose duration the policy cannot predict from its
observation.

**Test:** drop PLAN rows and retrain.
**Result: 9.80× → 8.50×.** The hold accounts for ~15% of the gap, not the bulk.
**Largely refuted.** The planned *paths* are the problem, not the dead time.

A sequence-safe variant (drop only rows where the robot did not move, keeping
the first frame of each pause) gave **8.33×** — agreement within 2% between two
filters that work differently, which is reassuring. That variant is the better
instrument because it leaves a continuous trajectory, so BC-RNN and ACT can use
it; the blunt phase drop would teach a chunked policy to step across a seam.

Selectivity at a 1e-4 rad threshold: removes 94.6% of PLAN rows, 13.8% of
REACH, 1.0% of GRASP. It targets the hold almost exactly without naming a phase.

---

## 4. The label bug, and why every number moved

On 2026-10-22, while checking whether the newer runs carried spawn jitter, I
found `contact_aware+anyteleop` held **31 episodes when 56 existed**.

**Cause:** hand-control normalises `args.mode` before logging it
(`kinova_leap_pick_place.py:561-575`). `contact_aware_w_anyteleop` becomes
`contact_aware_teleop` with the retargeter moved to a separate variable that
`run_config` never records. `logs.run_mode()` trusted the logged mode over the
directory name, so every `*_w_anyteleop_r0*` run from r03–r09 was relabelled
**dexpilot** — 25 episodes sitting in the manifest under the wrong key, and a
spurious `contact_aware+dexpilot` cell of exactly that size.

**Fix:** prefer the logged mode only when it is at least as specific as the
directory name. Regression test added. Method-label conflicts went 17 → 0.

**My error in finding it:** I first checked for `run_config` events by querying
only the 31 episodes already in the manifest, concluded "0 of 0 contact-aware
runs log jitter," and reported that the contact-aware arm had no spawn jitter.
It did. Absence from the manifest is not absence from disk.

### What the corrected data changed

| Measurement | v3 (31 eps) | v4 (41 matched) |
|---|---|---|
| BC-MLP gap | 3.6× | **9.8×** |
| Divergence, arm | 1.74× | **2.29×** |
| Divergence, hand | 1.86× | **3.05×** |
| Divergence, weighted | 2.10× | **3.09×** |
| Variance, hand | 0.80× (better) | **1.12× (worse)** |
| Arm/hand loss split | 7.86× / 2.60× | **20.60× / 8.06×** |

Everything got **larger**, and the one metric that favoured contact-aware
flipped. The "certification stabilises the squeeze" reading rested entirely on
that 0.80× and should now be dropped.

---

## 5. Why the October runs differ from September

The 9.8× figure raised a question: the newly-recovered episodes are much longer
(median 1564 vs 928 steps). Why?

**RRT solve time went from 5.5 ms to ~8000 ms.** But the planner did not get
slower — the *route* changed:

| | linear | rrt | linear_unchecked |
|---|---|---|---|
| September | **79%** | 9% | 0% |
| October | 0% | **74%** | **26%** |

In September the straight line from hand-over pose to pregrasp was collision-free
79% of the time, so `planner.plan()` never ran; the 5.5 ms was just the
`_edge_free` check returning True. The code comment anticipated this misreading
exactly: *"a fast plan now means 'the line was clear', not 'RRT was quick'."*
I fell for it before checking.

In October the line is blocked essentially always, so the real planner runs —
and 26% of the time it *fails*, and the arm executes an **unchecked straight
line** anyway. Those episodes contain motion that was never validated, and they
are in the training data.

### What caused the routing change — ruled out so far

- **Spawn jitter: NOT the cause.** October runs *without* jitter (r01–r02) show
  the same 75% rrt / 25% unchecked split as the jittered ones.
- **The gelatin `solref` change: NOT the cause.** It was stripped from
  `[0.012, 1.0]` to the rigid default, but the routing collapsed for **all four
  objects**, not just that one.

Still open. A uniform collapse points at something global in the collision
setup — the clearance matrix, endpoint grace, or the finger set it is built
over. `036_wood_block` was unpinned from a tripod to a 4-contact grasp in the
uncommitted working tree, which is the live candidate, but confirming it needs
a replay against the committed September code, not log analysis.

**There are no commits after 2026-09-15.** The October runs used a working tree
with 631 uncommitted insertions. That is the single biggest reproducibility
problem in this project.

---

## 6. Final results

All on `data/v4_still` (stationary rows collapsed, 41 matched episodes per
arm, AnyTeleop fixed, 8-fold CV over episodes):

| Algorithm | baseline | contact-aware | gap | Cohen d |
|---|---|---|---|---|
| BC-MLP | 0.0828 | 0.6893 | **8.3×** | −15.18 |
| BC-RNN | 0.0724 | 0.7364 | **10.2×** | −10.31 |
| ACT | 0.1747 | 0.5235 | **3.0×** | −10.74 |

**Arm/hand split** (BC-MLP, same dataset):

| Block | baseline | contact-aware | ratio |
|---|---|---|---|
| Pooled (23) | 0.0811 | 0.6833 | 8.42× |
| **Arm (7)** | 0.0371 | 0.6387 | **17.23×** (d = −23.08) |
| Hand (16) | 0.1004 | 0.7028 | 7.00× |

**Demonstration quality**, measured on the raw demos with no policy involved:

| Metric (lower better) | baseline | contact-aware | ratio |
|---|---|---|---|
| Action divergence, arm | 0.202 | 0.430 | 2.12× worse |
| Action divergence, hand | 0.105 | 0.305 | 2.91× worse |
| …(H−t) weighted | 0.273 | 0.816 | **2.99× worse** |
| Action variance, arm | 0.0080 | 0.0247 | 3.07× worse |
| Action variance, hand | 0.0059 | 0.0110 | 1.86× worse |
| State diversity | 5.35 | 6.42 | 1.20× higher |

Every metric now favours baseline — including hand variance, which was the one
exception on v3 (0.80×) and is now 1.86× worse. **The "certification stabilises
the squeeze" reading is dead.** What survives is the *relative* structure: the
arm is hurt roughly twice as badly as the hand (17.2× vs 7.0× in loss), and
state diversity remains the one thing contact-aware buys.

### Conclusions, in descending confidence

**1. Shared autonomy traded demonstration learnability for task success, here.
Measured on both sides of the chain.** Task success 94% vs 69%; demonstrations
2.3–3.1× worse by action divergence; policies 3–10× worse across three
architectures. Robust to: architecture, dataset version, episode-length
matching, object-mix balancing, neighbour count, and seed.

**2. The damage is in the arm, not the certified grasp.** Arm/hand loss split
20.6× vs 8.1×. The arm labels were never reconstructed, which closes the
obvious "your labels are bad" objection — the reconstruction affected only the
fingers, and the fingers are the better half.

**3. The cause is in the post-hand-over segment, but is not yet identified.**
Not the dead-time hold (removing it recovers only ~15%). Not the finger labels
(arm-only made it worse). Not sample size, not grip force. The planned paths
themselves are hard to learn, and *why* is open.

**4. Methodologically: measure the data, not just the policy.** The
demonstration metric predicted the policy result before training. That
agreement is the strongest evidence in the project, and it is the transferable
lesson — a data-quality metric computable from logs would have flagged this
pipeline before a single policy was trained.

### What this is NOT evidence for

- Closed-loop success is **0% for both conditions**, so the offline comparison
  is the only discriminating one. It does not predict deployment performance.
- One operator, one task family, one simulator.
- The October-vs-September routing change is unexplained, so part of the v4 gap
  may be an artifact of whatever changed in the uncommitted working tree.
- The causal mechanism is localised, not established.

---

## 7. What to do next

1. **Find the routing change.** `git stash` the working tree, replay one
   lock-in, see whether `_edge_free` returns True again. This is now the
   blocking question — it determines whether the v4 numbers measure the
   pipeline or an uncommitted regression.
2. **Commit the working tree.** 631 uncommitted insertions is why (1) is hard.
3. **Deterministic-planner arm.** Fix the RRT seed or post-process to a
   smoothed path, keep everything else identical. If arm loss falls toward
   baseline while task success holds at 94%, the mechanism is confirmed.
4. **Log the lock-in timestamp explicitly** — currently reconstructed from
   events, which makes a central quantity an inference.
5. The 26% `linear_unchecked` episodes should probably be excluded from
   training data, or at least flagged: they contain motion no collision check
   ever approved.
