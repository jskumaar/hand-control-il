# Speaker notes — Learning from Shared-Autonomy Teleoperation Data

Written for: you, presenting. Not for distribution.

Deck: `docs/IL_shared_autonomy.pptx` (23 slides, ~28 min).
Numbers trace to `runs/demo_quality.json`, `runs/exp_v3/summary.json`,
`runs/exp_v3_rnn/summary.json`, `runs/armonly.json`.

---

## Part 0 — The papers, in one place

Read this once before the talk. The Q&A depends more on knowing these than on
knowing the slides.

### The three that carry the argument

**Belkhale, Cui & Sadigh — "Data Quality in Imitation Learning", NeurIPS 2023
(arXiv:2306.02437).**
Your single most important citation; the headline metric is theirs. They argue
dataset quality is governed by two properties:
- *action divergence* — mismatch between the expert's and the learned policy's
  action at a state. In practice: do demonstrations disagree at states the
  policy cannot distinguish?
- *transition diversity* — noise in the **environment**, for a given
  state-action. **Not** demonstrator noise. People get this wrong constantly;
  do not be one of them.

Theorem 4.1 bounds the state-distribution shift by a sum of per-timestep action
divergences weighted by `(H−t)` — time-to-go. Early divergence propagates for
the rest of the episode. That is why your weighted row (2.10×) is worse than
your unweighted ones (1.74×, 1.86×).

Three findings you lean on:
1. "State diversity is not always beneficial" — and in their robomimic
   measurements state diversity is **uncorrelated** with policy success.
2. There is a **tradeoff**: buying coverage from a noisier demonstrator can cost
   more in action divergence than it buys in coverage.
3. They explicitly note prior work "aim[s] to maximize state diversity through
   shared control" while leaving the action side unexamined. That sentence
   describes your pipeline, written three years before your experiment.

Honest limitation they state: their single-timestep action-variance metric does
**not** capture long-horizon multimodality well — the "Worse" robomimic subsets
had *lower* measured action variance despite being qualitatively more
multimodal. If someone asks whether your metric could miss something, this is
the honest answer: yes, and the authors say so.

**Mandlekar et al. — "What Matters in Learning from Offline Human
Demonstrations for Robot Manipulation" (robomimic), CoRL 2021
(arXiv:2108.03298).**
Six algorithms, five sim + three real tasks, three data sources: MG
(machine-generated), PH (proficient human, 200 demos, 1 operator), MH
(multi-human, 300 demos, 6 operators).

The quote: *"results are lower for MH datasets compared to PH datasets, even
though the MH datasets contain 100 more demos (300 demos vs. 200 demos). This
most likely stems from the presence of suboptimal and multimodal data."*
**More data, worse policy.** That is your slide-3 argument in one sentence.

Proficiency split (MH partitioned into Better/Okay/Worse, 2 operators and 100
demos each): BC-RNN on Square falls **58 → 36 → 12 → 2**; on Can **96 → 56 → 40
→ 22**. Algorithm and volume fixed.

Two nuances to hold in reserve:
- Adding 100 "worse" demos: most algorithms decline, but **BC-RNN uniformly
  improves**. So "bad data hurts" is partly an architecture interaction.
- Quality still binds: BC-RNN on 200 mixed Square demos reaches 55–74% vs
  **84%** on 200 PH demos.
- **Lesson L4** is directly useful to you: 49–88% *relative* performance drops
  from adding seemingly-benign proprioception to the observation space. That is
  independent corroboration of your `corr(action, qpos) = 0.91` shortcut, from
  2021.

**Mandlekar, Garrett, Xu & Fox — "Human-in-the-Loop Task and Motion Planning
for Imitation Learning" (HITL-TAMP), CoRL 2023, PMLR v229.**
Note: **CoRL, not RSS.** TAMP handles the free-space motion; the human
teleoperates the contact-rich segment; a queueing system lets one operator serve
several robots.

Throughput: Coffee 431 vs 168 demos in 10 min; Square Broad 747 vs 166.
Policy success (per-user 10-min data): Coffee 90.7% vs 24.4%; Square Broad 80.0%
vs 1.2%.

Two things about it that matter more than the headline:
1. **BC trains only on the human portions.** TAMP segments are excised, never
   imitation targets. And **TAMP still runs at test time** — deployment uses "a
   TAMP-gated control loop that is identical to the handoff logic," with the
   policy substituted for the human. So they sidestep the mixed-provenance
   problem rather than solving it.
2. **Appendix L**, which is your slide 6. Verbatim: *"Surprisingly, TAMP-gating
   makes policies trained on the data comparable to HITL-TAMP data, but data
   collection still involves significantly higher operator time."* Conventional
   data + test-time gating reaches parity on several tasks (Three Piece
   Assembly 55.3% ungated → 96.0% gated, vs 100% for HITL-TAMP). Their own
   Appendix H attributes the advantage to **horizon reduction**: "the average
   human segment is small compared to the entire trajectory length."

Exception worth knowing: Tool Hang is the one task where HITL-TAMP still wins
after gating (80.7% vs 60.0 ± 19.6%).

### The theory

**Ross & Bagnell — "Efficient Reductions for Imitation Learning", AISTATS 2010.**
Theorem 2.1: `J(π̂) ≤ J(π*) + T²ε`, where ε is the expected **0-1 loss** under
the **expert's own** state distribution — i.e. ordinary BC training error. The
bound is on expected T-step **cost**, not mistake count, and it is **tight**
(they construct the matching system). Independently, **Rajaraman et al.
(NeurIPS 2020)** prove an information-theoretic lower bound of ~|S|H²/N that
holds even for a deterministic expert and even with active querying. So the
quadratic term is fundamental to the offline setting, not an artifact of loose
analysis.

**Ross, Gordon & Bagnell — DAgger, AISTATS 2011.** Do **not** say DAgger gives
O(εT) unconditionally. It is `uTε_N`, where `u` bounds the expert's cost-to-go
gap, and the paper admits **"in the worst case, u could be O(T)"** — in which
case there is no asymptotic gain over BC. Also `ε` means something different
here: `ε_N` is the best-in-hindsight loss under the **learner's** aggregated
distributions, not under `d_π*`.

**Laskey et al. — DART, CoRL 2017.** Noise injected into the **supervisor's**
policy at collection time, so the supervisor demonstrates recovery. +62% over BC
on grasping in clutter; 3× faster than DAgger in computation (**not** 280% — that
figure is a misquote in circulation).

Critically, DART does **not** support "optimal demonstrations hurt BC." The
paper says plainly that noise injection *"will offer no improvement if the robot
can represent the supervisor perfectly and collect sufficient data"*, and
Proposition 4.1 bounds a **covariate-shift KL**, not task cost. The actions in
DART are still expert actions — expert responses to perturbed states. The
defensible framing is about **coverage**, not optimality: narrow,
perfectly-executed demonstrations are information-poor about recovery.

**ACT (Zhao, Kumar, Levine & Finn, RSS 2023)** gives you the rebuttal to noise
injection for your setting: *"for fine manipulation, such noise injection can
directly lead to task failure, reducing the dexterity of teleoperation system."*
Also relevant: action chunking "reduces the effective horizon of the task by
k-fold, mitigating compounding errors", and they use a CVAE because "given the
same observation, a human can use different trajectories to solve the task."

### The shared-autonomy line

**Sha et al., arXiv:2603.17016 (2026).** Residual RL copilot trained in sim
against a kNN human surrogate. The strongest affirmative evidence: matched
*attempts* 18/20 grasp and 11/20 insertion vs 7/20 and 1/20 unassisted; matched
*successes* 19/20 and 9/20 vs 6/20 and 0/20. Their stated mechanism is almost
your hypothesis verbatim — demos "more consistent in alignment, contact timing,
and approach strategy." Caveat: one task, 5 participants, and they never
disambiguate whether the logged action is the human's raw input or the blended
copilot output.

**AnyTeleop (Qin et al., RSS 2023, arXiv:2307.04577).** Your own baseline, and
it already makes the smoothness version of your argument: *"(i) The collected
trajectory is more smooth, which means that the state-action pairs are more
consistent and easier to be consumed by the network. (ii) ... guarantees no
self-collision."* They validated it by training DAPG on data from two teleop
systems, 3 seeds × 100 trials. This is why your claim must be about
**certification**, not smoothness.

**PATO (Dass et al., RSS 2023).** Learned assistive policy executes repetitive
segments, queries the human under uncertainty. Reduced operator mental load,
improved throughput, one operator for several robots. **Never trains a
comparison policy** — all 23 of its 2025+ citers were enumerated and none closes
that gap. Cite it as "assistance for the operator, not validated for the
learner."

**Sirius (Liu et al., RSS 2023; IJRR 44(10–11):1727–1742, **2024**).** Note the
year — IJRR 2024, published online 2024-10-23, DOI 10.1177/02783649241273901.
Human-gated shared control at deployment with intervention-weighted BC over
**four** provenance classes: demonstrations, autonomous actions, interventions,
and *pre-interventions* (downweighted, because the human intervened precisely
where things were going wrong). +8% sim, +27% real, 3× faster convergence.
The point for you: it *presupposes* provenance is a clean recoverable
per-timestep flag and never asks whether the logged action during an assisted
segment is imitable.

**IWR (Mandlekar et al., arXiv:2012.06733, 2020).** Upweight human-intervention
samples, because interventions begin near bottlenecks. An **arXiv preprint**,
not a conference paper.

**BRIDGE (arXiv:2606.26603, 2026).** Your slide 8 and slide 17. Handheld
collection captures *observed* actions; teleoperation captures *desired*
actions. Mixing them is a label-semantics conflict, not a quality gradient:
*"The Naive Mix baseline fails because mixing observed and desired labels leads
to imprecise manipulation."* NIST pulley **44.0% → 0%**; battery **10.0% →
0.0%**. Their fix is a state-gated mixture of diffusion experts (+36.7%).
Reweighting could not have fixed this — which is exactly the gap in the
IWR/Sirius framing.

**DATAFARM (arXiv:2609.12316, 2026).** *"Raw TAMP trajectories provide
surprisingly little benefit when used to fine-tune pretrained VLA models,
despite successfully solving the target tasks."* 8.3% vs 61.7% for human teleop,
attributed to behavioural distribution mismatch. Task success ≠ learnable data,
independently.

### The near-misses you must cite and distinguish

**FABCO (arXiv:2602.15351, NAIST, 2026).** The closest published analogue to
your argument. Estimates **feasibility under the robot's dynamics model** and
feeds it to the demonstrator during collection via visual + haptic cues.
*"FABCO improves imitation learning performance by more than 3.2 times compared
to the case without feasibility feedback"*, n=15 participants.
**Your distinction:** FABCO certifies *kinematic/dynamic reproducibility* —
"don't demonstrate what the robot can't execute." FGT certifies *wrench
feasibility of a grasp* — "don't demonstrate grasps that can't resist task
wrenches." Related, genuinely different, and you must say so before a reviewer
does.

**BlenDAgger (arXiv:2609.37599, CMU — Smith, Sun, Admoni, Erickson).** Blended
shared control for interactive IL. +30 or more percentage points over HG-DAgger,
and they attribute it to **57% smoother transitions** between policy and human
control, plus 14% higher trajectory similarity to training data. Note: this is
your own institution. **Your distinction:** they measure smoothness *within* a
transition; you measure *dispersion across* a demonstration set in where the
transition occurs.

**LfDI (arXiv:2610.17298, IROS 2026, McGill/Mila).** *"Learning from delayed
expert interventions is overall less effective than learning from non-delayed
expert interventions."* **Your distinction:** delay is a systematic bias; yours
is variance. Both are about hand-over timing, so concede the territory and hold
the narrower claim.

**TAPs (arXiv:2512.04960, KTH).** Teleoperation augmentation primitives — axis
locks, perching waypoints, pose anchors; i.e. virtual fixtures. Opens with the
thesis sentence of your talk: *"What an operator can demonstrate bounds what
imitation learning can learn."* Square 0.372→0.531, real unscrewing 38%→67%.
Clean design — the same demos are replayed with primitives inserted, so
unassisted is an identical control. **But** the primitives are recorded in the
demonstration and triggered by the policy at test time, so it carries the
Appendix L confound, and its own claim is the weaker *"primitives that help the
operator do not hurt the policy."*

**Wrench-ACT (arXiv:2609.37552, 2026).** Your slide 20. *"Cross-condition
ablations show that the bilateral data collection interface and the wrench
action space each contribute independently to performance."* The factorial
separation of interface from architecture that nobody in the shared-autonomy
data line has done.

**HATS (arXiv:2606.16491, 2026).** Agent-assisted vs expert dual-human vs
full-agent, 7 tasks, 100 demos/task held constant. Full-agent 29.8, dual-human
70.3, **HATS 71.3** — parity, not gain. The most broadly-scoped of the three
matched comparisons, and it is the one that found nothing.

### The contested metric literature (slide 19)

All four are **short single-author preprints or workshop posters, not
peer-reviewed.** Say so when you cite them.

- **arXiv:2610.17427, "Smoothness Ranks Skill, Not Success"** (NeurIPS 2026
  RoboPAD workshop poster, Bassi). Audits RINSE's spectral arc length under
  episode-length controls: SAL AUROC 0.952 on can/mh, but **episode length alone
  gets 0.957**; in a duration-matched band SAL falls to **0.628** while raw mean
  jerk — the baseline RINSE argues against — rises to **0.896**; on can/mg,
  where every episode is exactly 150 steps, SAL is at chance (0.454).
- **arXiv:2606.10229, "What Demonstration Curation Metrics Do to Your Policy"**
  (Bedi). Detection and benefit decouple: best-AUROC metric (0.804) → **worst**
  policy (13.3%); a 0.638-AUROC metric nearly matched oracle (90.0% vs 93.3%).
- **arXiv:2606.05588, "Auditing Demonstration Curation Metrics"** (Bedi).
  *"Structural errors... are invisible to every action-only metric we test, and
  two of them are inverted: they score defective demonstrations as higher
  quality."* If your variance measure becomes a gate or scorer, this is the
  adversarial evaluation a knowledgeable reviewer will ask for.
- **RINSE, arXiv:2604.23000** — the target of the above. Proposes SAL and
  Trajectory-Envelope Distance for curation; robomimic +16% on 1/6 the data.
  Mention it only as the thing being audited.

**DemInf (Hejna et al., RSS 2025, arXiv:2502.08623)** is the peer-reviewed
counterweight: k-NN mutual-information estimation over VAE embeddings, partitions
datasets in agreement with human expert scores, and **filtering by it yields
5–10% improvement on robomimic** plus real ALOHA/Franka gains. The closest thing
to a validated predictive data-quality metric. All 56 of its 2025+ citers were
enumerated: **none applies it to mixed-provenance or assisted-vs-unassisted
data.**

---

## Part 1 — Per-slide notes, questions, and answers

### Slide 1 — Title
**Say:** Name, date, topic. "This is about what the teleoperation pipeline's data
is worth to a learner — and the answer surprised me."
**Timing:** 15 s. Do not linger.

---

### Slide 2 — The premise this work tests
**Say:** Open with the TAPs sentence — it is the thesis of the whole area. Then:
assistance exists so an operator can demonstrate what they otherwise could not.
The field's claim is that this also makes the *data* better. AnyTeleop — which is
one of the two retargeters in this pipeline — states the learning consequence
outright. So the premise is not a straw man; it is what the systems I build on
believe.

End on the question, and say you will answer it in two halves.

**Q: Isn't this obvious? Better demonstrations must be better data.**
A: That intuition is exactly what the talk tests, and the literature is split on
it. robomimic found 300 demonstrations training a worse policy than 200, because
the 300 came from six operators. "Better" for the task and "better" for the
learner are different objectives, and I will show they can conflict.

**Q: Why does AnyTeleop appear as both a baseline and a cited claim?**
A: Deliberately. It is the retargeter held fixed across both conditions, so the
hand mapping cannot explain any difference I report. And its authors already
argued the smoothness version of this hypothesis, which is why my claim has to be
about certification rather than smoothness.

---

### Slide 3 — Why demonstration consistency should matter
**Say:** Three results, increasing in concreteness. Ross & Bagnell: BC error
compounds quadratically in the horizon, and that is tight — there is a matching
lower bound, so it is not an artifact of loose analysis. Belkhale's Theorem 4.1
adds *where* it hurts: the `(H−t)` weight means early divergence propagates
longest. Then robomimic makes it empirical: more data, worse policy, because of
who collected it.

Land the proficiency numbers — 58 → 36 → 12 → 2 — slowly. Algorithm fixed,
volume fixed, only the operator changed.

**Q: Isn't T²ε a worst case that rarely binds in practice?**
A: It is tight in the sense that a matching construction exists, and Rajaraman et
al. prove an information-theoretic lower bound of order H² that holds even for a
deterministic expert and even if the learner may actively query. So you cannot
escape the horizon dependence by better analysis — only by changing the setting,
which is what DAgger and action chunking do.

**Q: Doesn't DAgger solve this?**
A: Conditionally. The bound is `uTε_N`, where `u` bounds the expert's cost-to-go
gap, and the paper itself notes `u` could be O(T) in the worst case — in which
case there is no asymptotic improvement. It also requires an interactive expert,
which a certified-grasp pipeline does not provide: the certificate is not an
oracle policy you can query at an arbitrary state.

**Q: robomimic is simulation. Does it transfer?**
A: Their Lesson L6 reports the results transfer to their real-world tasks. But
the more relevant point for this talk is that my own result is measured on my own
data; robomimic is motivation, not evidence for my claim.

---

### Slide 4 — The hypothesis, as a chain of two links
**Say:** This is the slide that organises everything after it. The hypothesis is
a chain, and chains fail link by link. Link 2→3 is among the best-supported
claims in IL, so I take it as given. Link 1→2 is asserted widely and tested
rarely — never for certification as the mechanism. So that is the one I measure,
and I measure it *before* training any policy, so a failure can be localised
rather than blamed on the learner.

Flag that you will come back and answer both links with numbers.

**Q: Why not just test end-to-end? If policies improve, who cares which link?**
A: Because the end-to-end test is what I ran first, and it gave a negative result
I could not interpret. A policy failing could mean bad data, a bad action space,
too little data, or a bug. Measuring the middle of the chain turns an
uninterpretable negative into a localised one — and in this case the data metric
*predicted* the policy result, which is much stronger evidence than either alone.

---

### Slide 5 — What the literature actually establishes
**Say:** This is the state of the evidence, counted rather than impressionistic.
At least four systems assert that assisted collection improves the data. Exactly
three run a matched assisted-vs-unassisted downstream comparison. Of those, two
show a gain — and the broadest, HATS at seven tasks with demo count held
constant, finds parity.

Be explicit about method: forward citation graph of nine seed papers, 2025+
citers enumerated.

**Say the tone out loud if needed:** this is not an attack on the field. It is a
claim that is expensive to test and cheap to assert, so it has been asserted.

**Q: Isn't "matched comparison" an unfairly strict bar?**
A: It is the bar the claim requires. If you say assisted data trains better
policies, you need two policies trained on assisted and unassisted data with
volume and task held constant. PATO, for instance, measures operator mental load
and throughput — both real contributions — but never trains a comparison policy,
so it cannot support the data-quality claim that people cite it for.

**Q: Are you sure you did not miss papers?**
A: Possibly, and one branch is knowingly incomplete: the robomimic citation
sweep hit a 1000-result API cap, so that branch is truncated. The other eight
seeds were enumerated fully. I would describe the count as a lower bound on
matched comparisons and a floor on assertion-without-test, not as exhaustive.

**Q: Sha et al. is a positive result with matched successes. Doesn't that settle it?**
A: It is the strongest affirmative evidence and I credit it on slide 18's
spirit. But: one task, five participants, and they never disambiguate whether the
logged action is the operator's raw input or the blended copilot output — which
is precisely the ambiguity slide 8 is about. It is good evidence, not settled
science.

---

### Slide 6 — The confound nobody has resolved
**Say:** HITL-TAMP is the paper closest to my claim, and its own appendix
complicates it. Read the quote aloud — "Surprisingly" is their word. Apply the
gating architecture at test time to policies trained on *conventional* data and
they rise to parity. Their Appendix H attributes the gain to horizon reduction.

So the field's strongest result for assisted collection may be a result about
test-time architecture. 52 citations, 40 from 2025 on, all enumerated — nobody
has revisited it.

Then the turn: FGT is absent at deployment. The policy runs alone, no
certificate, no solver. So this pipeline can separate the two explanations.

**Q: Are you claiming HITL-TAMP is wrong?**
A: No. Their throughput result is large and uncontested, and their policy
numbers are real. I am claiming the *interpretation* that assisted data is
intrinsically better is not established by their experiment, and they say as
much in the appendix. That is a reading of their paper, not a refutation.

**Q: If gating is what helps, why not just use gating?**
A: Legitimate, and it may be the right engineering answer. But it changes the
deployment requirement: you then need the planner at test time, which for a
wrench-certified grasp means solving an NLP in the loop on hardware. The reason
to want better *data* is that data is free at deployment.

**Q: Tool Hang still favoured HITL-TAMP after gating. Doesn't that save the claim?**
A: It is the one task that does, at 80.7% vs 60.0 ± 19.6%, and I should say so
rather than omit it. One task out of several where the effect survives is
suggestive, not decisive — and the variance on the gated conventional arm is
large.

---

### Slide 7 — The theory predicts assistance could hurt
**Say:** Now the uncomfortable part. Belkhale et al. define two properties that
govern data quality, and emphasise that transition diversity is *environment*
noise, not demonstrator noise — people routinely misread it.

Then the three findings. State diversity, which is what shared control is
usually praised for providing, is uncorrelated with success in their
measurements. And there is a tradeoff: coverage bought from a noisier
demonstrator can cost more than it buys. Then read their sentence about prior
work maximising state diversity through shared control — written before my
experiment, describing my pipeline.

Say plainly: the hypothesis had a stated prior against it, and I did not notice
until after I had the result.

**Q: Isn't more state coverage necessary for generalisation?**
A: Necessary but not sufficient, and the way you obtain it matters. Their point
— and mine — is that coverage obtained by *admitting more valid solutions*
raises action divergence at the states you already had. You want coverage from
varied initial conditions, not from varied responses to the same condition.

**Q: Could you get the coverage without the divergence?**
A: That is exactly what the two-week experiment tests. If the divergence comes
from hand-over timing rather than from certification itself, then fixing the
timing keeps the coverage and removes the inconsistency. I will know in two
weeks.

---

### Slide 8 — What the shared-autonomy literature does not ask
**Say:** Everyone in this area treats provenance as a clean label and
reweights by it. Sirius partitions into four classes, including pre-interventions
which it downweights because that is where things were going wrong. IWR upweights
interventions because they start at bottlenecks. The DAgger variants ask *when*
to query, then take the human's action as ground truth.

None asks whether the logged action during an assisted segment is an imitable
human command at all.

Then BRIDGE, the one paper on the mechanism, with the numbers: 44% → 0%. Mixing
observed and desired action labels is not a quality gradient you can reweight
away — it is a semantic conflict.

**Connect it to your own system if asked:** this is concretely your
`q_grasp_hold` problem. During the certified grasp phase the commanded target
was never logged, so the finger labels had to be reconstructed. That is the same
class of problem BRIDGE names.

**Q: Isn't this just a logging bug on your side?**
A: Partly, and I fixed the logging. But the deeper issue survives correct
logging: even with the exact commanded target recorded, the question of whether
a *solver's* command is something a policy should imitate as if it were human
intent is a modelling question, not a logging question.

**Q: Why not just drop the assisted segments, like HITL-TAMP does?**
A: That is a real option and it works for them. It requires the assistance to be
present at deployment, which returns us to slide 6. And in my case the certified
grasp *is* the interesting part of the task — excising it leaves you imitating
the reach.

---

### Slide 9 — The pipeline
**Say:** Brief, factual. 23 actuated joints, MuJoCo, RoboCasa counter, four YCB
objects. The certificate: a grasp commits only when the contact set can resist
the task wrench, with the squeeze magnitude from the same solve, warm-started by
the retargeted operator pose. Operator drives Approach; the certificate gates
entry to Pick.

Emphasise two design facts that matter for the rest of the talk: the retargeter
is held fixed in every comparison, and **assistance is absent at deployment.**

**Timing:** 90 s. This is the audience's orientation slide, not a contribution
slide.

**Q: Is the grasp solved online or precomputed?**
A: Online, warm-started from the operator's retargeted pose, which is what makes
it a shared-autonomy system rather than a grasp library.

**Q: Why four objects?**
A: They span the geometry cases that matter for a certificate — a box, a
cylinder, a sphere, and a mug whose handle makes a canonical grasp
ill-defined. The mug is consistently the hardest, which is useful.

---

### Slide 10 — Pipeline results
**Say:** Certification works on the task metric. 94% vs 69%, p = 0.0068. And the
advantage grows with difficulty — under the stiffness ablation the baseline falls
to 56% while contact-aware holds at 100%, which independently replicates the FGT
paper's Table II on data collected here.

Then the setup line for the pivot: **so link 1 holds, on the task metric.**

**Q: Is the success criterion the same for both conditions?**
A: Yes — identical criteria from the trial logger, applied post hoc to logs from
both conditions. No per-condition tuning.

**Q: Could the baseline operator just be less practised?**
A: The runs are interleaved rather than blocked by condition, so a learning
effect would have to apply to both. But I will concede this is a single-operator
study and operator skill is not randomised — which is a limitation of the whole
dataset, not just this slide.

---

### Slide 11 — How learnability is measured
**Say:** Four commitments, each adopted because the weaker version gave a wrong
answer. K-fold CV over episodes, because a single held-out split holds out one or
two episodes and the loss mostly reports *which*. Episode-level splits, never
per-sample, because windows ten steps apart are near-duplicates. Matched counts
and a fixed retargeter. Rollouts of at least 2500 steps, because at 400 every
policy scores zero.

Then the one line about earlier errors being in backup, and **move on**. Do not
elaborate unless asked.

**Q: Why is validation loss the learnability metric rather than task success?**
A: Because closed-loop success is currently zero for both conditions, so it has
no discriminative power — it would report a tie that means nothing. Validation
loss on held-out *episodes* is the sharpest available signal, and the sign of the
effect is what matters, not its absolute value. I would rather report a
discriminative proxy honestly than a non-discriminative headline.

**Q: Closed-loop success is zero? Then is any of this actionable?**
A: It is actionable for the data-collection decision, which is the decision in
front of me in the next two weeks. It is not yet actionable for deployment, and I
say so on the summary slide. The honest status is: the plumbing is verified, the
labels are exact, the policies fit the data, and the closed-loop gap is a
learning problem rather than an infrastructure one.

**Q: 31 episodes per condition is small.**
A: Agreed, and it is why the metric choice matters so much. K-fold over episodes
makes the metric about 12× more stable than a single split, and the effect sizes
here are d = −8 to −10, so the comparison is not noise-limited. Whether it
*generalises* beyond this task family is a separate and open question.

---

### Slide 12 — Link 1 → 2 fails  ★ the central result
**Say:** Slow down here. This is the slide the talk exists for.

The contact-aware demonstrations are measurably worse *as data* — before any
policy is trained, by the metric from a peer-reviewed NeurIPS paper. Action
divergence 1.74× worse on the arm, 1.86× on the hand, and 2.10× once you apply
Belkhale's time-to-go weight, because the divergence is concentrated where it
compounds longest.

Explain the metric in one sentence: disagreement between demonstrated actions at
the nearest *cross-episode* observation neighbours. Say why cross-episode —
within an episode the nearest neighbours are consecutive frames, so you would be
measuring the control timestep, not the disagreement.

Then the robustness line, which is what makes it a result rather than an
artifact: stable across seeds, insensitive to `k` over 4/8/16, and with the
object mix equalised at six per object per arm it is 1.98× ± 0.14 on the arm and
2.12× ± 0.07 on the hand, with all eight draws above 1.

Be honest about the one row that goes the other way: hand action *variance*
favours contact-aware at 0.80×. That is consistent with the certificate
stabilising the squeeze while destabilising the approach — and it is the kind of
detail I would want to know if I were in the audience.

**Q: Isn't this circular? You chose a metric that captures what you found.**
A: Fair challenge, and the answer is the ordering. The metric and its threshold
come from Belkhale et al., published 2023, and the direction of its prediction —
lower divergence means better policies — was fixed before I computed anything.
I then checked whether the metric's prediction matched the policy result, and it
did, in sign and roughly in magnitude. A metric chosen to fit would not also have
had to survive that test.

**Q: Could the divergence be an artifact of your observation space?**
A: It could in principle — divergence is measured at observation neighbours, so a
weaker observation makes more things look like neighbours. Two things argue
against it. The observation is identical across conditions, so it cannot create a
*difference*. And the object identity, size and mass are in the observation
precisely because an earlier version lacked them and the matched-neighbour
analysis showed only 32% of baseline neighbours shared the query's object.

**Q: 31 vs 41 episodes — did you subsample, and does that matter?**
A: Matched at 31, with the surplus baseline episodes dropped at random. And the
object-balanced re-run at six per object per arm is the stronger version of the
same control: it equalises both count and composition, and the effect gets
slightly *larger*, not smaller.

**Q: Why is hand variance better for contact-aware if divergence is worse?**
A: Because they measure different things. Variance is the spread of the
neighbours' own actions; divergence is their distance from the query's action.
The reading I find most plausible is that the certificate makes the final squeeze
very repeatable — which lowers local variance — while making the *approach to*
that squeeze less repeatable, which raises divergence. That is also consistent
with slide 15's finding that the trajectories agree at the grasp.

---

### Slide 13 — The failure is the predicted tradeoff
**Say:** This is why the result is more than a negative. Certification bought
22% more state diversity and paid roughly 2× in action divergence. That is
precisely the exchange Belkhale et al. identified — and it is the wrong direction
on both counts, because state diversity is the quantity they find *uncorrelated*
with success and action divergence is the one that predicts it.

Then the mechanism in plain language, and this is the sentence to land: the
pipeline optimised the metric that does not matter and degraded the one that
does — not through a bug, but because **certification admits multiple valid
solutions, and the operator picks a different one each time.**

**Q: So is certification simply a bad idea?**
A: No, and I would resist that reading. Certification is a clear win on the task
— 94% vs 69%, and more under difficulty. What this says is that task success and
demonstration learnability are different objectives, and optimising the first
does not give you the second for free. The fix is probably not to abandon
certification but to constrain *which* valid solution gets chosen.

**Q: Is 22% more state diversity not worth having?**
A: Only if you can get it without the divergence cost. Belkhale's measurement is
that state diversity on its own does not predict policy success, so 22% more of
it is not obviously worth 2× the divergence. That is an empirical trade, and the
fixed-timing experiment is how I would price it.

---

### Slide 14 — Link 2 → 3 holds
**Say:** And the data metric's prediction came true. BC-MLP: 0.0968 vs 0.3477,
3.6×. BC-RNN replicates at 3.9×. Cohen's d of −8.08 and −9.80, both p < 0.0001.
Remind them these are losses, so lower is better — the baseline data is easier to
fit on held-out episodes.

Then the inversion, stated cleanly: contact-aware wins the task 94% to 69% and
loses the learning comparison by 3.6 to 3.9×. **That is the exchange rate.**

Note the architecture point: BC-RNN is a recurrent policy, and robomimic found
BC-RNN is the architecture that best absorbs low-quality data. It did not rescue
this. So the effect is not an artifact of a weak policy class.

**Q: Why not use a stronger policy — diffusion, ACT?**
A: ACT and implicit BC are implemented and queued; they are on the deferred list
precisely because I do not want to claim the inversion is universal before
testing it. What I can say now is that it holds across a feedforward and a
recurrent policy, including the one robomimic identifies as most tolerant of
mixed-quality data.

**Q: Could the contact-aware data just be higher-entropy, so higher loss is expected?**
A: That is close to being the same claim as the result — higher conditional
entropy of action given observation *is* what action divergence measures. The
reason it is a finding rather than a tautology is that it was not supposed to go
this way: the hypothesis, and the literature's expectation, was that certified
grasps would make the data *more* predictable.

**Q: Is validation loss comparable across datasets with different statistics?**
A: Both conditions are normalised with a single normaliser fit over the pooled
data, so the scales are shared. That is a deliberate choice — per-condition
normalisation would rescale each arm by its own spread and make the comparison
meaningless.

---

### Slide 15 — Localising the cause: hand-over variance
**Say:** So where does the divergence live? Not everywhere. The demonstrations
are consistent early in the episode and consistent again at the grasp —
contact-aware is actually *more* consistent at the grasp on three of four
objects. They diverge in between, which is exactly where the certificate takes
over.

The switch point itself is the variable quantity: lock-in at a median 76% through
the trial, with a p10–p90 spread of 28 percentage points. The solver's readiness
is not synchronised to the operator's intent, so the hand-over lands somewhere
different every episode.

Why that is the right candidate: a policy conditioned on observation alone cannot
infer how far through the hand-over it is. The same observation maps to different
actions depending on an unobserved phase — which is state aliasing in the sense of
the phase-conditioning work.

**Flag the confidence level explicitly:** this is localised, not causally
established. The fixed-timing arm is what would establish it.

**Q: Why not just add phase to the observation?**
A: It is on the deferred list and the phase-conditioning paper reports 45% → 90%
with FiLM conditioning, so it is promising. Two cautions. The phase label leaks
information a policy will not have at deployment unless you can compute the phase
online. And in this dataset the phase stream is partly a function of the teleop
mode rather than of physics, so a naive one-hot would encode which condition the
data came from — which is why I dropped it from the observation earlier.

**Q: Couldn't the variance be the operator rather than the solver?**
A: Both contribute, and I cannot fully separate them with the current logging,
because lock-in time is reconstructed from events rather than recorded directly.
That is the first fix in the two-week plan. The fixed-timing arm then forces the
solver side to be constant.

**Q: Is 28 points a lot?**
A: Relative to what a Markovian policy can infer, yes — it means the same
observation can sit before or after hand-over depending on the episode. In
absolute terms it is about a quarter of the trial.

---

### Slide 16 — Alternative explanations
**Say:** Three plausible causes, each tested, each refuted. The first was my own
hypothesis: that the repaired finger labels were driving the gap. Arm-only
training refuted it by making the gap *more* than twice as large — 3.19× to
8.82×. Worth saying out loud that this was my idea and the measurement killed it.

More data: pooling 16 to 63 episodes moved the loss from 0.00174 to 0.00166 —
nothing. Grip force: the squeeze already reaches 1210 N against the demos' 168 N,
and these objects need between 0.23 and 3.6 N to hold statically. Grip force is
not the binding constraint; reach is.

Each refutation narrows the cause toward the hand-over.

**Q: Arm-only making it worse seems strange. Why would removing the hand hurt?**
A: It tells you the arm is where the inconsistency lives, which is the opposite
of what I expected — I had assumed the certified squeeze was the messy part.
Removing the fingers removes the one signal that was *consistent* and leaves the
divergent reach, so the ratio gets worse. It is consistent with slide 15: agree
at the grasp, disagree on the way there.

**Q: 16 to 63 episodes is not much data. Would 500 change it?**
A: Possibly, and I cannot rule it out. But the scaling behaviour is the relevant
signal: a near-flat response over a 4× increase in data suggests the limit is not
sample size. And the phase-conditioning paper argues explicitly that state
aliasing is not fixed by more data, which is the same mechanism I am proposing.

---

### Slide 17 — This result is not an outlier
**Say:** Two 2026 papers find assistance-provenance effects that harm learning.
BRIDGE: 44% to 0% from mixing action-label semantics. DATAFARM: raw planner
trajectories give "surprisingly little benefit despite successfully solving the
target tasks" — 8.3% against 61.7% for human teleoperation.

Then the observation that matters: **neither cites the shared-autonomy
data-collection line.** The two literatures have not met, so the line asserting
the benefit has not had to absorb this counterevidence.

And the positioning: my contribution is to run the comparison inside a pipeline
where assistance is absent at deployment, so there is no gating confound.

**Q: Those are preprints. How much weight do they carry?**
A: Less than a peer-reviewed result, and I would not build the argument on them
alone — which is why the argument rests on Belkhale and on my own measurements.
They matter as convergent evidence from independent setups, and because their
mechanisms are specific and checkable rather than vague.

**Q: DATAFARM is about VLA fine-tuning. Is that comparable to your BC setting?**
A: Not directly, and I should say so. The transferable part is the dissociation —
data that solves the task is not automatically data that trains well. The
mechanism they name, behavioural distribution mismatch against a pretrained
model, does not apply to a policy trained from scratch.

---

### Slide 18 — What is novel, and what is not
**Say:** Stated plainly, because three of these were claimed in the last twelve
months and the audience may know them better than I do.

FABCO is the closest: feasibility-certified collection, 3.2×, n=15. They certify
kinematic reproducibility — don't demonstrate what the robot cannot execute. I
certify wrench capacity — don't demonstrate grasps that cannot resist task
wrenches. Related, genuinely different, and I would rather draw the distinction
myself than have it drawn for me.

BlenDAgger — note, from this institution — measures hand-over smoothness and gets
+30 points. LfDI shows delayed interventions are less effective. The
absolute-versus-delta action-space finding is established, including robomimic's
49–88% drops from adding proprioception back in 2021.

What is open: hand-over *variance* as distinct from bias, delay or
within-transition smoothness; wrench-capacity certification as the mechanism; and
a clean decomposition of data quality from test-time architecture.

**Q: If BlenDAgger and LfDI cover hand-over, what is left?**
A: A narrow but real distinction. BlenDAgger measures how smooth a single
transition is; LfDI measures a systematic delay — a bias. Neither measures
*dispersion across a demonstration set* in where the hand-over occurs. That is
the quantity that predicts BC difficulty in my data, and it is computable from
logs before training.

**Q: Isn't "variance versus bias" splitting hairs?**
A: It matters for what you do about it. A bias you correct by shifting the
hand-over point; a variance you correct by synchronising it. Those are different
engineering interventions, and the second is the one my two-week experiment tests.

**Q: Why not frame the action-space finding as a contribution?**
A: Because it is a rediscovery, and the 2026 action-space paper has 13,000
rollouts behind it against my handful. What I have that they do not is a cleaner
mechanistic account — the 0.91 correlation and policies commanding 13–39% of
needed travel — so it earns a backup slide and a sentence, not a section.

---

### Slide 19 — A caveat about the metrics themselves
**Say:** I want to undercut my own measure before someone else does.

The demonstration-curation literature is actively contested. One audit reports
spectral arc length at AUROC 0.952 while episode length *alone* reaches 0.957 —
so the metric may largely be a length proxy. Duration-matched, it falls to 0.628
while plain mean jerk rises to 0.896. A second finds detection accuracy and
policy benefit decouple: the best-detecting metric gave the worst policy.

Say clearly: **these audits are single-author preprints, not peer-reviewed.**
Weight them accordingly. But the episode-length confound is concrete and
checkable, so I checked it — my smoothness numbers survive duration matching,
d = 0.80 to 0.91.

Then the important distinction: the headline claim does not rest on smoothness. It
rests on action divergence, from a peer-reviewed NeurIPS paper, and that is also
the metric whose prediction the policy result confirmed.

**Q: Why include this at all if it weakens you?**
A: Because it is the first thing a knowledgeable reviewer would raise, and
because the distinction between my contested metrics and my load-bearing one is
genuinely favourable to me. Omitting it would make the talk look naive about its
own evidence.

**Q: Have you run the adversarial testbed from those audits?**
A: No, and if the variance measure becomes a curation gate rather than a
diagnostic, that is the evaluation it needs. Right now it is a diagnostic — I use
it to localise a cause, not to filter data — so the audit's criticism applies less
directly. If I turn it into a filter, the criticism applies fully.

**Q: Could action divergence have the same length confound?**
A: It is less exposed by construction: it is computed per-query at observation
neighbours and averaged, rather than integrated along a trajectory, so a longer
episode contributes more queries but not a larger per-query value. I have not run
a duration-matched control on divergence specifically, and I should.

---

### Slide 20 — The design the next experiment should copy
**Say:** Wrench-ACT does the decomposition properly: a factorial ablation
separating the bilateral collection interface from the wrench action space, and
each contributes independently. That is precisely what HITL-TAMP's Appendix L
left open, done in an adjacent force-controlled setting.

Applied here: interface factor is certified versus unassisted collection;
architecture factor is absolute versus delta action space, and phase-conditioned
versus observation-only. Crossing them separates "the data is better" from "the
policy class handles this data better," which no shared-autonomy data paper has
done.

**Q: Why not run the full factorial in the two weeks?**
A: Because collection is the bottleneck and the window is two weeks. The full
cross is four interface-by-architecture cells on top of three collection arms,
and I would rather have one clean falsifiable result than four underpowered ones.
The architecture factor can run later on data already collected; the interface
factor cannot.

---

### Slide 21 — Plan: the next two weeks
**Say:** Week one is collection, with the logging fixes. Equal counts per
condition and per object, ±5 cm spawn jitter, varied initial arm pose, and the
GRASP-phase target, gains and live gamma all logged. And the lock-in timestamp
logged explicitly rather than reconstructed — without that, the central quantity
in this talk is an inference.

Week two is the decisive experiment: three arms, with the third being
contact-aware with fixed-timing lock-in. The prediction is falsifiable — if
hand-over variance is the mechanism, arm three closes the learnability gap while
keeping the task-success advantage. If it does not, the mechanism is something
else and I will say so.

Point out that this is also the HITL-TAMP decomposition: assistance held
constant, its timing varied.

**Q: Is fixed-timing lock-in even achievable? Won't the solver sometimes not be ready?**
A: That is the main risk. The honest version is a fixed *budget* rather than a
fixed instant — commit at a predetermined point and fall back if the certificate
has not converged — and I will need to report how often the fallback fires. If it
fires often, the arm is contaminated and I will say so.

**Q: What if the fixed-timing arm hurts task success?**
A: Then I have priced the trade, which is itself the result: certification helps
the task through exactly the flexibility that hurts the data. That is a more
interesting finding than a clean win.

**Q: Two weeks is tight. What if collection slips?**
A: The collection is the part I control, and the analysis pipeline already exists
and is tested — `run_experiment.py` and `test_demo_quality.py` both run
end-to-end today. If collection slips, the fallback is to run the comparison on
fewer episodes per cell and report it as underpowered rather than to extend.

---

### Slide 22 — Summary
**Say:** Three claims, descending confidence, and say the confidence levels out
loud.

One, measured: shared autonomy traded demonstration learnability for task
success here — demonstrations 1.74 to 2.12× worse by action divergence, policies
3.6 to 3.9× worse by cross-validated loss, against a 94% to 69% task win. The
data metric predicted the policy result, which is what makes it more than an
anecdote.

Two, localised but not causally established: hand-over variance is the likely
exchange rate. The fixed-timing arm tests it.

Three, about the field: the consensus rests on three matched comparisons. At
least four systems assert the benefit without testing it, the closest paper has
an unresolved confound, and the counterevidence sits in a literature that does not
cite this one.

**Q: What is the one-sentence takeaway?**
A: Assistance that helps the operator succeed does not automatically produce data
that helps a policy learn, and the two can trade against each other — so if you
are building a shared-autonomy system to collect training data, measure the data,
not just the success rate.

**Q: Would you build this pipeline again?**
A: Yes, and I would instrument the hand-over from day one. The certificate is a
clear task-level win; what I would change is treating demonstration learnability
as a design objective rather than an assumed by-product.

---

### Slide 23 — Thank you
**Say:** Keep it short and let it land. Eleven months, from a grasp recommender
that collided with its own object to a certified system with a measured answer
about what its data is worth. Thank the lab for run time, hardware, and the
arguments that changed your mind — several of which are on the slides.

Then the last line, and mean it: the negative result is the one you are proudest
of, because it was harder to measure and it is the one worth passing on.

**Timing:** 45 s. Do not add material here.

---

## Part 2 — Cross-cutting questions

These can arrive at any point. Have them ready.

**Q: Your closed-loop success is zero for both conditions. Why should anyone
believe the offline comparison means anything?**
A: Because the offline comparison is the one that discriminates, and it does so
with d = −8 to −10 on episode-level cross-validation. Zero-versus-zero is an
uninformative tie, not evidence of equivalence. I would not claim the offline gap
predicts a deployment gap — I would claim it predicts which data is easier to fit,
which is the question in front of me when deciding what to collect next.

**Q: One operator, one task family, one simulator. How general is this?**
A: Not demonstrated to be general at all, and I would not defend it as such. What
I would defend: inside this setting the effect is large, robust to the obvious
controls, replicated across two policy architectures, and predicted in advance by
an independent peer-reviewed metric. The right next step is a second task family,
not a stronger claim about this one.

**Q: Could this all be a bug?**
A: It could, and four previous results in this project were bugs — which is why I
now measure both ends of the chain. The specific reassurance here is that the
demonstration metric and the policy metric are computed by different code paths
over different representations and agree in sign and roughly in magnitude. A bug
that produced that coincidence would have to be unusually cooperative.

**Q: Why is this worth presenting if the hypothesis failed?**
A: Because the hypothesis failing is the informative outcome. The field has
asserted the opposite roughly four times for every once it has tested it, and the
one paper closest to the claim has an unresolved confound in its own appendix. A
matched comparison in a pipeline without that confound is worth having regardless
of which way it comes out — and this one came out against the consensus, which
makes it more worth having, not less.

**Q: What would change your mind?**
A: The fixed-timing arm closing the gap would confirm the mechanism. The
fixed-timing arm *not* closing it, with divergence staying high, would mean
certification itself raises divergence and my mechanism is wrong. And a second
task family showing the opposite sign would mean the effect is specific to this
setup. All three are cheap to run relative to the claim they would settle.

---

## Part 3 — Numbers to have memorised

| Quantity | Value | Source |
|---|---|---|
| Task success, baseline vs contact-aware | 69% vs 94%, p = 0.0068 | pipeline eval |
| Stiffness ablation (s0p4) | baseline 56%, contact-aware 100% | replicates FGT Table II |
| Action divergence, arm | 0.208 → 0.362 (1.74×) | `runs/demo_quality.json` |
| Action divergence, hand | 0.108 → 0.200 (1.86×) | same |
| (H−t) weighted divergence | 0.278 → 0.584 (2.10×) | same |
| Object-balanced re-run | arm 1.98× ± 0.14, hand 2.12× ± 0.07, 8/8 draws > 1 | same |
| State diversity | 5.38 → 6.53 (1.22×) | same |
| BC-MLP, 8-fold CV, 12 seeds | 0.0968 vs 0.3477, d = −8.08 | `runs/exp_v3/summary.json` |
| BC-RNN, 8-fold CV, 6 seeds | 0.0851 vs 0.3362, d = −9.80 | `runs/exp_v3_rnn/summary.json` |
| Arm-only ablation | 3.19× → 8.82× | `runs/armonly.json` |
| Lock-in timing | median 76%, p10–p90 spread 28 points | trace analysis |
| Episodes per condition | 31, AnyTeleop both arms | `data/v3/manifest.json` |
| Data scaling | 16 → 63 eps: 0.00174 → 0.00166 | pooling test |
| Squeeze vs demand | 1210 N available, 168 N in demos, 0.23–3.6 N needed | force analysis |
