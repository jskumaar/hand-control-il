# Results slides — paste into Google Slides

Written for: you, building the deck by hand in Google Slides.

**How to use.** One `---` block per slide. Paste the title into the title box
and the bullets into the body box; Google Slides keeps the indent levels. The
`[FIGURE: ...]` lines name a PNG in `docs/figs/` to insert — Insert → Image →
Upload from computer.

**Status of these numbers.** Everything below is the **v3** dataset: 31 matched
episodes per arm, AnyTeleop held fixed. A label bug found on 2026-10-22 meant
the contact-aware cell was under-filled — 31 of the 56 episodes that actually
exist — because `run_config.mode` logs the *normalised* mode and silently
relabelled runs r03–r09 as dexpilot. The re-extraction (`data/v4`) is running.
**Expect the ratios to move; expect the direction to hold.** Slides marked
⚠️ are the ones most likely to change.

---

## SLIDE 1 — Title

**Where the Variance Is**
Contact-aware vs baseline demonstrations

Suresh Kumaar Jayaraman · AiPEX Lab · Carnegie Mellon University
October 2026

---

## SLIDE 2 — The question

**Does shared autonomy produce better data for imitation learning?**

- The pipeline certifies a grasp is wrench-feasible before committing to it
  - Task success: **94% vs 69%** (p = 0.0068)
  - Under the stiffness ablation the gap widens: 100% vs 56%
- So the demonstrations are *better at the task*
- The question this deck answers: are they better *to learn from*?

Speaker note: the hypothesis was "certified grasps → better demos → better
policies." Link 1 (certification improves the task) is established. This deck
tests links 2 and 3.

---

## SLIDE 3 — How learnability is measured ⚠️ explain this slowly

**Validation loss on held-out EPISODES, 8-fold cross-validated**

- Split the 31 episodes into 8 folds; train on 7, test on the held-out one
  - **Episodes, never samples.** Windows ten steps apart in one episode are
    near-duplicates; a per-sample split made losses look 25–85× better than
    they were
- Record the best validation loss across 60 epochs, average over the 8 folds
  - That is one number per seed; the reported figure is the mean over seeds
- Matched episode counts, retargeter held fixed, so pipeline is the only variable
- Actions are 23 joint position targets in radians, z-scored with one
  normaliser fit over both conditions pooled

**What the loss function is:** MSE for BC-MLP and BC-RNN; L1 + KL for ACT
(a CVAE). So losses are **not comparable across algorithms** — only the ratio
within each pair.

---

## SLIDE 4 — Result 1: policies trained on contact-aware data are worse

[FIGURE: fig1_learnability.png]

- BC-MLP **3.6×** · BC-RNN **3.9×** · ACT **1.7×**
- Cohen d = −8.08, −9.80, −8.16. All p < 0.0001
- Three architectures, same sign — not an artifact of one policy class
- ACT narrows the gap most, as chunking plus a CVAE latent should. It does not
  close it
- robomimic identifies BC-RNN as the architecture most tolerant of
  mixed-quality data. It did not rescue this

---

## SLIDE 5 — Result 2: the gap is in the ARM

[FIGURE: fig2_armhand.png]

- Same training runs, validation loss split by joint block
- **Arm (7 joints): 7.9× and 7.5×.  Hand (16 joints): 2.6× and 3.2×**
- Both significant (d = −10.3, −9.0), but the arm gap is about 3× the hand gap

**Why this matters:** the finger labels were partly reconstructed (the
GRASP-phase PD target was not logged), so "your labels are bad" is the obvious
objection. The **arm** labels were never reconstructed — recovery is exact
there — and the arm is where the gap is.

Also note baseline's own structure inverts: there the hand is the harder block
(0.121 vs 0.047). Certification made the arm the hard part.

---

## SLIDE 6 — Result 3: it is visible in the data, before training

[FIGURE: fig4_divergence.png]

- **Action divergence** — do demonstrations disagree at states the policy
  cannot tell apart? Measured as action distance among nearest *cross-episode*
  observation neighbours. Lower is better
- Arm **1.74×** worse, hand **1.86×** worse, **2.10×** worse once weighted by
  time-to-go (early divergence compounds longest)
- **Hand action variance is 0.80× — the one metric contact-aware wins.** The
  certified squeeze is more repeatable
- **State diversity +22%** — but Belkhale et al. find state diversity
  *uncorrelated* with policy success

Robustness: stable across seeds, insensitive to k (4/8/16 → 1.74–1.97×), and
with object mix equalised, 1.98× ± 0.14 (arm), all 8 draws above 1.

---

## SLIDE 7 — Diversity vs divergence: the distinction that matters

**Two things get called "variance" and they pull opposite ways**

- **State diversity** — how much of the observation space the demos visit
  - Good. It is what lets a policy handle a new situation
  - You get it from varied object positions and start poses
- **Action divergence** — at the *same* observation, do demos disagree?
  - Bad, always. No policy can be right about all of them at once; regression
    averages them into something valid for none

**Contact-aware bought +22% of the first and paid ~2× in the second.** It
optimised the metric that does not predict success and degraded the one that
does.

The usable test: is the variation **observable**? Object position is
(`obj_in_palm` is in the 78-D observation), so two demos reaching differently
for differently-placed objects are both right. Variation the policy cannot
see in its input is pure divergence.

---

## SLIDE 8 — Where in the trial they disagree

[FIGURE: band.png]

- Every episode resampled to 100 normalised-time points; spread taken across
  episodes at each point
- Both conditions **start tight** (~4.5 cm)
- Baseline **re-converges** by the end (9.0 cm); contact-aware **stays wide**
  (18.8 cm)
- Dotted lines: contact-aware hand-over times, spanning **2% to 31%** of the trial

⚠️ **Caveat to state out loud:** on the wood block the palm-spread ratio is only
1.04×, while the loss gap is 3.4×. So *spatial* spread is not what drives the
learnability gap — this figure shows **where** things differ, not **why**. The
loss and divergence numbers are the evidence.

---

## SLIDE 9 — What happens after hand-over

**Over half of each contact-aware episode is planner output — and it is in the
training labels**

| | baseline | contact-aware |
|---|---|---|
| RRT-driven share of frames | 0% | **59% mean** (24–86%) |
| Hand-over event | none | lock-in, 2–31% through trial |

- The extraction reads `control_phase` only to pick a label-recovery method —
  it never filters rows, so PLAN and REACH frames are training labels
- RRT is sampling-based: same start and goal can give a different path
- Cross-episode palm spread *within* the REACH segment: 14–21 cm, with
  **end-points scattering 20–33 cm**

⚠️ **Honest limit:** RRT's start AND goal both differ between episodes (the
operator hands over from a different pose; the certificate picks a different
grasp). So this does **not** yet isolate planner randomness from goal
variation. Measured: the segment after hand-over is where demos disagree most.
Not measured: why, within that segment.

---

## SLIDE 10 — Why shared autonomy made this worse

**Assistance did reduce variability — in the part it directly controls**

- Where it helped: hand action variance **0.80×**. The certified squeeze is
  more repeatable
- Where it hurt: arm variance 1.80× worse, arm validation loss ~7.5× worse
- The certificate hands control away at an unpredictable moment, and what it
  hands to produces a different path each time

**Variability was not removed — it was relocated from the fingers to the arm.**

---

## SLIDE 11 — The theory predicted this

**Belkhale, Cui & Sadigh, "Data Quality in Imitation Learning," NeurIPS 2023**

- Two properties govern data quality: **action divergence** (demos disagreeing
  at indistinguishable states) and **transition diversity** (noise in the
  *environment*, not the demonstrator)
- They find **state diversity is not correlated with policy success**
- They identify the tradeoff: coverage bought from a noisier demonstrator can
  cost more in divergence than it buys
- Their words on prior work: it "aim[s] to maximize state diversity **through
  shared control**" while leaving the action side unexamined

That sentence describes this pipeline, written before the experiment.

---

## SLIDE 12 — What this is NOT evidence for

**Stated plainly, because the data has limits**

- Closed-loop success is **0% for both conditions** — the offline comparison is
  the only one that discriminates. It does not predict deployment performance
- One operator, one task family, one simulator
- 31 episodes per arm ⚠️ and the contact-aware cell was under-filled by a label
  bug; re-running on the corrected 56
- The causal claim (planner non-determinism) is **not** established — only
  localised to the post-hand-over segment
- Failure-mode trajectories shown earlier are "abandoned" trials from outside
  the comparison set

---

## SLIDE 13 — What to do next

**Two experiments, in order of cost**

1. **Deterministic planner arm** — fix the RRT seed or post-process to a
   smoothed path, keep everything else identical
   - Prediction: arm loss falls toward baseline, task success stays at 94%
   - If it does not, the mechanism is something else
2. **Log the lock-in timestamp explicitly** — currently reconstructed from
   events, which makes the central quantity an inference
3. Collect both arms with matched spawn jitter and matched seeds (partly done
   already — runs r03–r09)

**Open question:** is a certificate that admits many valid paths inherently at
odds with learnable demonstrations?

---

## SLIDE 14 — Summary

**One sentence: the certificate improved the grasp and degraded the reach.**

- **Measured** — demos 1.74–2.12× worse by action divergence; policies
  1.7–3.9× worse across three architectures; task success 94% vs 69%
- **Localised** — arm ~7.5–7.9× vs hand 2.6–3.2×. Hand variance is the single
  metric favouring contact-aware
- **Not yet explained** — the post-hand-over segment is where demos diverge,
  but the cause within it is open

The data metric predicted the policy result. That agreement, between two
independent measurements, is what makes this more than an anecdote.
