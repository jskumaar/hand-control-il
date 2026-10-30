#!/usr/bin/env python3
"""Build the IL-from-shared-autonomy-data slide deck.

Format is inherited from docs/Format.pptx (CMU AiPEX template), measured rather
than guessed:

    slide        13.333 x 7.5 in (12192000 x 6858000 EMU)
    font         Arial everywhere
    title        sz 2933, bold, at L=609600 T=268224 W=10972800 H=707136
    red rule     C41230, w=12700, full width at y=999744
    body         sz 1600 bold (level 0), 1733 (level 1)
    insets       lIns=tIns=rIns=bIns=0 on every text frame
    accent       C41230 (title-slide background, and the rule)

Deviations from the template, per the author's explicit instructions:

  * text is BLACK only -- the template's red titles and 555555 sub-bullets are
    overridden
  * no autoshape outlines or fills behind text (template already uses noFill,
    but we assert it)
  * no footer notes at the bottom of the slide; citations instead sit as a
    smaller-font block on the slide where that work is discussed
  * title slide is name/date/topic only

Every number in this deck traces to a file in this repo:

    runs/demo_quality.json      demonstration-quality metrics (test_demo_quality)
    runs/exp_v3/summary.json    BC-MLP learnability comparison
    runs/exp_v3_rnn/            BC-RNN replication
    runs/armonly.json           arm-only ablation

Usage:  uv run python scripts/make_deck.py --out docs/IL_shared_autonomy.pptx
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Pt

# ---------------------------------------------------------------- template spec

EMU_IN = 914400
SLIDE_W = 12192000
SLIDE_H = 6858000

CMU_RED = RGBColor(0xC4, 0x12, 0x30)
BLACK = RGBColor(0x00, 0x00, 0x00)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FONT = "Arial"

# Geometry lifted verbatim from the template's slide 2.
TITLE_L, TITLE_T = 609600, 268224
TITLE_W, TITLE_H = 10972800, 707136
RULE_Y = 999744
BODY_T = 1219200
BODY_H = 5364480

# Type scale. The template uses 2933/1600/1733; we keep the title and level-0
# size and set level 1 slightly below level 0 so the hierarchy reads as
# hierarchy without the template's colour cue (which black-only text removes).
SZ_TITLE = 2800
SZ_L0 = 1600
SZ_L1 = 1400
SZ_CITE = 1000
SZ_KICKER = 1500

BULLET = "•"


def _style(run, size, bold=False, color=BLACK, italic=False):
    f = run.font
    f.name = FONT
    f.size = Pt(size / 100)
    f.bold = bold
    f.italic = italic
    f.color.rgb = color


def _textbox(slide, left, top, width, height):
    """A text box with no fill and no outline, insets zeroed like the template."""
    tb = slide.shapes.add_textbox(Emu(left), Emu(top), Emu(width), Emu(height))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tb, tf


def _para(tf, first_para_used: list, text, size, bold=False, level=0,
          bullet=False, color=BLACK, space_before=0, italic=False):
    """Append a paragraph. `first_para_used` tracks the implicit empty one."""
    if not first_para_used:
        p = tf.paragraphs[0]
        first_para_used.append(True)
    else:
        p = tf.add_paragraph()
    p.level = level
    if space_before:
        p.space_before = Pt(space_before)
    label = f"{BULLET}  {text}" if bullet else text
    r = p.add_run()
    r.text = label
    _style(r, size, bold=bold, color=color, italic=italic)
    return p


def _blank(prs):
    """A slide from the blank layout, with an explicit white background."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = WHITE
    return slide


def _rule(slide, y=RULE_Y, left=TITLE_L, width=TITLE_W):
    """The template's thin CMU-red horizontal rule. A line, not a box."""
    from pptx.enum.shapes import MSO_CONNECTOR

    ln = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Emu(left), Emu(y), Emu(left + width), Emu(y))
    ln.line.color.rgb = CMU_RED
    ln.line.width = Emu(12700)
    return ln


def content(prs, title, blocks, cite=None, kicker=None):
    """A standard content slide.

    blocks: list of (text, level) or (text, level, bold). Level 0 is bold by
            default, level 1 plain, matching the template's hierarchy.
    cite:   list of citation strings, rendered small at the foot of the body
            block (NOT as a slide footer).
    kicker: one line directly under the rule, for the slide's claim.
    """
    slide = _blank(prs)

    _, tf = _textbox(slide, TITLE_L, TITLE_T, TITLE_W, TITLE_H)
    used: list = []
    _para(tf, used, title, SZ_TITLE, bold=True)

    _rule(slide)

    # Reserve room at the foot of the body for citations, if any.
    n_cite = len(cite or ())
    cite_h = int(n_cite * 0.26 * EMU_IN) + (int(0.14 * EMU_IN) if n_cite else 0)
    body_h = BODY_H - cite_h

    _, tf = _textbox(slide, TITLE_L, BODY_T, TITLE_W, body_h)
    used = []
    if kicker:
        _para(tf, used, kicker, SZ_KICKER, bold=True)
    for blk in blocks:
        text, level = blk[0], blk[1]
        bold = blk[2] if len(blk) > 2 else (level == 0)
        _para(tf, used, text, SZ_L0 if level == 0 else SZ_L1,
              bold=bold, level=level, bullet=True,
              space_before=8 if level == 0 else 2)

    if n_cite:
        _, tf = _textbox(slide, TITLE_L, BODY_T + body_h + int(0.10 * EMU_IN),
                         TITLE_W, cite_h)
        used = []
        for c in cite:
            _para(tf, used, c, SZ_CITE)
    return slide


def table_slide(prs, title, headers, rows, notes=(), cite=None, kicker=None,
                col_w=None):
    """A content slide whose body is a monospaced-by-tabstop table.

    python-pptx tables bring their own borders and fills, which the author asked
    to avoid, so the table is laid out as separate text boxes per column --
    plain text, no shapes.
    """
    slide = _blank(prs)
    _, tf = _textbox(slide, TITLE_L, TITLE_T, TITLE_W, TITLE_H)
    used: list = []
    _para(tf, used, title, SZ_TITLE, bold=True)
    _rule(slide)

    top = BODY_T
    if kicker:
        _, tf = _textbox(slide, TITLE_L, top, TITLE_W, int(0.42 * EMU_IN))
        used = []
        _para(tf, used, kicker, SZ_KICKER, bold=True)
        top += int(0.56 * EMU_IN)

    ncol = len(headers)
    if col_w is None:
        first = int(TITLE_W * 0.40)
        rest = (TITLE_W - first) // max(ncol - 1, 1)
        col_w = [first] + [rest] * (ncol - 1)

    row_h = int(0.34 * EMU_IN)
    header_h = int(0.40 * EMU_IN)

    x = TITLE_L
    for ci, (h, w) in enumerate(zip(headers, col_w)):
        _, tf = _textbox(slide, x, top, w, header_h)
        used = []
        p = _para(tf, used, h, SZ_L1, bold=True)
        if ci:
            p.alignment = PP_ALIGN.RIGHT
        x += w

    y = top + header_h + int(0.06 * EMU_IN)
    for r in rows:
        x = TITLE_L
        for ci, (cell, w) in enumerate(zip(r, col_w)):
            _, tf = _textbox(slide, x, y, w, row_h)
            used = []
            bold = ci > 0 and cell.startswith("*")
            p = _para(tf, used, cell.lstrip("*"), SZ_L1, bold=bold)
            if ci:
                p.alignment = PP_ALIGN.RIGHT
            x += w
        y += row_h

    y += int(0.16 * EMU_IN)
    if notes:
        _, tf = _textbox(slide, TITLE_L, y, TITLE_W,
                         int(len(notes) * 0.30 * EMU_IN))
        used = []
        for n in notes:
            _para(tf, used, n, SZ_L1, bullet=True, space_before=3)
        y += int(len(notes) * 0.30 * EMU_IN)

    if cite:
        _, tf = _textbox(slide, TITLE_L, max(y + int(0.12 * EMU_IN),
                                             BODY_T + BODY_H - int(len(cite) * 0.26 * EMU_IN)),
                         TITLE_W, int(len(cite) * 0.28 * EMU_IN))
        used = []
        for c in cite:
            _para(tf, used, c, SZ_CITE)
    return slide


# ---------------------------------------------------------------- the deck


def build(out: Path) -> Path:
    prs = Presentation()
    prs.slide_width = Emu(SLIDE_W)
    prs.slide_height = Emu(SLIDE_H)

    # ---- 1. title -------------------------------------------------------
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = CMU_RED

    _, tf = _textbox(slide, 670560, 2100000, 10850880, 2000000)
    used: list = []
    _para(tf, used, "Learning from Shared-Autonomy", 4000, bold=True, color=WHITE)
    _para(tf, used, "Teleoperation Data", 4000, bold=True, color=WHITE)

    _, tf = _textbox(slide, 670560, 4500000, 10850880, 1400000)
    used = []
    _para(tf, used, "Suresh Kumaar Jayaraman", 2000, color=WHITE, space_before=0)
    _para(tf, used, "17 October 2026", 1600, color=WHITE, space_before=10)

    # ---- 2. the premise -------------------------------------------------
    content(
        prs, "The premise",
        kicker="Assistance lets an operator demonstrate what they otherwise could not.",
        blocks=[
            ("The field's claim: assistance also makes the DATA better", 0),
            ("AnyTeleop: smoother retargeting gives “state-action pairs "
             "[that] are more consistent and easier to be consumed by the "
             "network” — validated by training DAPG on data from two "
             "teleop systems.", 1),
            ("Our pipeline", 0),
            ("FGT certifies wrench feasibility before a grasp commits: the "
             "operator supplies preference, the solver supplies feasibility.", 1),
            ("The question", 0),
            ("Certified grasps raise task success. Do they also produce "
             "demonstrations that are better to learn from?", 1),
        ],
        cite=["van Haastregt et al., Hybrid Imitation Learning (TAPs), arXiv:2512.04960, 2025. Qin et al., AnyTeleop, RSS 2023."])

    # ---- 3. why consistency should matter -------------------------------
    content(
        prs, "Why consistency should matter",
        kicker="BC error compounds over the horizon — and early error compounds most.",
        blocks=[
            ("Compounding error is quadratic in the horizon, and the bound is tight", 0),
            ("J(π̂) ≤ J(π*) + T²ε, for supervised "
             "error ε under the expert's own state distribution. A matching "
             "lower bound holds even for a deterministic expert.", 1),
            ("Early inconsistency is weighted more heavily", 0),
            ("The distribution-shift bound carries an (H−t) time-to-go "
             "weight: divergence near the start propagates for the rest of the "
             "episode.", 1),
            ("More data does not rescue inconsistent data", 0),
            ("robomimic: 300 multi-human demos train a WORSE policy than 200 "
             "proficient-human ones — six operators instead of one.", 1),
            ("Across proficiency tiers, algorithm and volume fixed: BC-RNN on "
             "Square falls 58 → 36 → 12 → 2.", 1),
        ],
        cite=["Ross & Bagnell, AISTATS 2010, Thm 2.1. Rajaraman et al., NeurIPS 2020 (lower bound).",
              "Belkhale, Cui & Sadigh, NeurIPS 2023, Thm 4.1. Mandlekar et al., robomimic, CoRL 2021."])

    # ---- 4. the chain ---------------------------------------------------
    content(
        prs, "The hypothesis as a chain",
        kicker="Certified grasps → better demonstrations → better policies.",
        blocks=[
            ("Link 2 → 3   well supported — taken as given", 0),
            ("robomimic, Belkhale et al., and the demonstration-curation "
             "literature all measure it. We do not re-litigate it.", 1),
            ("Link 1 → 2   asserted widely, tested rarely", 0),
            ("And never for certification as the mechanism. This is the link we "
             "measure — before any policy is trained.", 1),
            ("Measuring both separately lets a failure be localised", 0),
            ("1 → 2: action divergence and variance on the demonstrations "
             "themselves.", 1),
            ("2 → 3: K-fold cross-validated BC loss, retargeter held fixed.", 1),
        ])

    # ---- 5. the counting slide -----------------------------------------
    table_slide(
        prs, "What the literature establishes",
        kicker="The consensus rests on three matched comparisons.",
        headers=["", "Count"],
        rows=[["Systems asserting assisted collection improves the data",
               "*≥ 4"],
              ["Matched assisted-vs-unassisted downstream policy comparisons",
               "*3"],
              ["… of those, showing a gain rather than parity", "2"]],
        notes=["All three matched comparisons are from 2026, each covering a single task family.",
               "The broadest (HATS: 7 tasks, 100 demos/task held constant) finds parity — 71.3 vs 70.3.",
               "PATO, NestDex and DexGrasp Copilot assert the data benefit without ever training a comparison policy.",
               "Method: forward citation graph of nine seed papers, all 2025+ citers enumerated."],
        cite=["PATO, RSS 2023. HITL-TAMP, CoRL 2023. Sha et al., arXiv:2603.17016. TAPs, arXiv:2512.04960. HATS, arXiv:2606.16491."])

    # ---- 6. the unresolved confound ------------------------------------
    content(
        prs, "The confound nobody has resolved",
        kicker="HITL-TAMP's own ablation credits test-time gating, not data quality.",
        blocks=[
            ("Appendix L, verbatim", 0),
            ("“Surprisingly, TAMP-gating makes policies trained on the data "
             "comparable to HITL-TAMP data, but data collection still involves "
             "significantly higher operator time.”", 1),
            ("Conventional unassisted data, gated at test time, reaches parity: "
             "Three Piece Assembly 55.3% → 96.0%, against 100%.", 1),
            ("Their Appendix H credits horizon reduction — the human segment "
             "is short relative to the trajectory.", 1),
            ("Still unaddressed", 0),
            ("52 citations, 40 of them from 2025 on, all enumerated: no "
             "follow-up and no journal version revisits it.", 1),
            ("Why it matters here: FGT is absent at deployment, so this pipeline "
             "can separate the two explanations.", 0),
        ],
        cite=["Mandlekar, Garrett, Xu & Fox, HITL-TAMP, CoRL 2023 (PMLR v229), Appendix L and Appendix H."])

    # ---- 7. the prior against us ---------------------------------------
    content(
        prs, "The theory predicts assistance could hurt",
        kicker="Shared control buys state coverage and pays in action consistency.",
        blocks=[
            ("Two properties govern data quality", 0),
            ("Action divergence — demonstrations disagreeing at states the "
             "policy cannot distinguish. Lower is better.", 1),
            ("Transition diversity — noise in the ENVIRONMENT for a given "
             "state-action, not noise in the demonstrator.", 1),
            ("Their findings about coverage", 0),
            ("State diversity is NOT correlated with policy success in their "
             "robomimic measurements.", 1),
            ("Coverage bought from a noisier demonstrator can cost more in "
             "action divergence than it buys.", 1),
            ("They note prior work maximises state diversity “through shared "
             "control” while leaving the action side unexamined.", 1),
            ("That sentence describes this pipeline, written before the experiment.", 0),
        ],
        cite=["Belkhale, Cui & Sadigh, Data Quality in Imitation Learning, NeurIPS 2023."])

    # ---- 8. the unasked question ---------------------------------------
    content(
        prs, "The unasked question",
        kicker="Everyone treats provenance as a clean label to reweight.",
        blocks=[
            ("Provenance as a reweighting problem", 0),
            ("Sirius partitions deployment data into four classes — demos, "
             "autonomous actions, interventions, pre-interventions — and "
             "reweights by class.", 1),
            ("IWR upweights interventions because they begin at bottlenecks. "
             "Both presuppose a clean per-timestep flag.", 1),
            ("None of them asks", 0),
            ("During an assisted segment, is the logged action an imitable human "
             "command at all?", 1),
            ("The one paper on the mechanism", 0),
            ("BRIDGE: handheld logs observed actions, teleoperation logs desired "
             "actions. Mixing them is a semantic conflict, not a quality "
             "gradient — NIST pulley 44.0% → 0%. Reweighting cannot fix it.", 1),
        ],
        cite=["Sirius, RSS 2023; IJRR 44(10–11):1727–1742, 2024. IWR, arXiv:2012.06733, 2020.",
              "ThriftyDAgger, CoRL 2021. LazyDAgger, CASE 2021. BRIDGE, arXiv:2606.26603, 2026."])

    # ---- 9. the pipeline ------------------------------------------------
    content(
        prs, "The pipeline: Feasible Grasp Teleoperation",
        kicker="The operator proposes; a wrench-feasibility certificate decides.",
        blocks=[
            ("Platform", 0),
            ("Kinova Gen3 (7 DOF) + LEAP hand (16 DOF) = 23 actuated joints. "
             "MuJoCo, RoboCasa counter scene, four YCB objects.", 1),
            ("What the certificate does", 0),
            ("A grasp commits only when the contact set can resist the task "
             "wrench; the squeeze magnitude γ* comes from the same solve, "
             "warm-started by the retargeted operator pose.", 1),
            ("Four phases — Approach, Pick, Transport, Place. The operator "
             "drives Approach; the certificate gates entry to Pick.", 1),
            ("Two design facts that matter below", 0),
            ("The retargeter is held FIXED in every comparison, so the hand "
             "mapping cannot explain a difference.", 1),
            ("Assistance is absent at deployment — the policy runs with no "
             "certificate and no solver.", 1),
        ])

    # ---- 10. pipeline results ------------------------------------------
    table_slide(
        prs, "Pipeline results: certification works",
        kicker="And the advantage grows as the task gets harder.",
        headers=["Condition", "Task success"],
        rows=[["Baseline teleoperation", "69%"],
              ["Contact-aware (FGT)", "*94%"]],
        notes=["Two-proportion test: p = 0.0068. Identical success criteria applied post hoc to both conditions.",
               "Under the object-stiffness ablation (solref 0.004): baseline 56%, contact-aware 100%.",
               "That difficulty reversal independently replicates the FGT paper's Table II, on data collected here.",
               "So link 1 of the chain holds — on the task metric."])

    # ---- 11. how we measured learnability ------------------------------
    content(
        prs, "How learnability is measured",
        kicker="Four commitments, each adopted because a weaker version gave a wrong answer.",
        blocks=[
            ("K-fold cross-validation over EPISODES", 0),
            ("A single held-out split holds out 1–2 episodes, so the loss "
             "largely reports which episode that was — seed sd of "
             "65–154% of the mean. K-fold is 12× more stable.", 1),
            ("Episode-level splits, never per-sample", 0),
            ("Windows ten timesteps apart in one episode are near-duplicates; a "
             "sample split put them on both sides.", 1),
            ("Matched episode counts and a fixed retargeter", 0),
            ("31 episodes per condition, AnyTeleop in both arms — neither "
             "volume nor the hand mapping can explain a difference.", 1),
            ("Closed-loop rollouts of at least 2500 steps", 0),
            ("At 400 steps every policy scores zero; the first real pick appears "
             "at 2500 and plateaus there.", 1),
            ("Earlier measurement errors, and what each invalidated, are in the backup.", 0, False),
        ])

    # ---- 12. THE RESULT: link 1->2 fails -------------------------------
    table_slide(
        prs, "Link 1 → 2 fails, by the field's own metric",
        kicker="The contact-aware demonstrations are worse as data — before any policy is trained.",
        headers=["Metric  (lower is better)", "Baseline", "Contact-aware", "Ratio"],
        rows=[["Action divergence, arm", "0.208", "0.362", "*1.74× worse"],
              ["Action divergence, hand", "0.108", "0.200", "*1.86× worse"],
              ["… (H−t) time-to-go weighted", "0.278", "0.584", "*2.10× worse"],
              ["Action variance, arm", "0.0076", "0.0137", "1.80× worse"],
              ["Action variance, hand", "0.0053", "0.0042", "0.80× better"],
              ["State diversity  (no implied direction)", "5.38", "6.53", "1.22× higher"]],
        notes=["Divergence: mean action disagreement among the k nearest CROSS-EPISODE observation neighbours. Within an episode the nearest neighbours are consecutive frames — that would report the control timestep, not disagreement.",
               "Robust: stable across seeds, insensitive to k (4/8/16 give 1.74–1.97×).",
               "Object mix equalised at 6 per object per arm: arm 1.98× ± 0.14, hand 2.12× ± 0.07, all 8 draws above 1.",
               "Hand variance is the one row favouring contact-aware — the certificate stabilises the squeeze while destabilising the approach."],
        cite=["Metric definitions follow Belkhale, Cui & Sadigh, NeurIPS 2023 (action divergence, action variance, state diversity)."])

    # ---- 13. why: exactly the predicted tradeoff -----------------------
    content(
        prs, "The failure is the predicted tradeoff",
        kicker="Certification bought coverage and paid in consistency.",
        blocks=[
            ("What it bought", 0),
            ("State diversity up 22% — the certificate admits a wider set of "
             "valid grasps, so demonstrations visit more of the space.", 1),
            ("What it cost", 0),
            ("Action divergence up ~2×: at observations a policy cannot tell "
             "apart, the demonstrations now disagree about what to do.", 1),
            ("Weighted by time-to-go the cost is 2.10× — concentrated "
             "where it compounds longest.", 1),
            ("Why that is the wrong trade", 0),
            ("State diversity is the quantity Belkhale et al. find UNCORRELATED "
             "with policy success; action divergence is the one that predicts it.", 1),
            ("Not a bug: certification admits multiple valid solutions, and the "
             "operator picks a different one each time.", 1),
        ],
        cite=["Belkhale, Cui & Sadigh, NeurIPS 2023 — state diversity uncorrelated with success; the coverage-vs-divergence tradeoff."])

    # ---- 14. link 2->3 holds -------------------------------------------
    table_slide(
        prs, "Link 2 → 3 holds — the data metric predicted the policy",
        kicker="Worse demonstrations trained worse policies, in the direction the metric implied.",
        headers=["Algorithm", "Baseline", "Contact-aware", "Gap"],
        rows=[["BC-MLP  (8-fold CV, 12 seeds)", "0.0968", "0.3477", "*3.6×"],
              ["BC-RNN  (8-fold CV, 6 seeds)", "0.0851", "0.3362", "*3.9×"]],
        notes=["Cohen d = −8.08 and −9.80, both p < 0.0001. The inversion is not architecture-specific.",
               "Validation loss, so LOWER is better: the baseline data is easier to fit on held-out episodes.",
               "robomimic identifies BC-RNN as the architecture most tolerant of mixed-quality data. It did not rescue this.",
               "The inversion: contact-aware wins the task 94% vs 69% and loses the learning comparison by 3.6–3.9×."])

    # ---- 15. cause: hand-over variance ---------------------------------
    content(
        prs, "Localising the cause: hand-over variance",
        kicker="They agree early and agree at the grasp. They disagree about when to switch.",
        blocks=[
            ("Where the trajectories diverge", 0),
            ("Consistent early, and consistent again at the grasp — "
             "contact-aware is MORE consistent at the grasp on three of four "
             "objects.", 1),
            ("Divergent in between, which is exactly where the certificate takes "
             "over.", 1),
            ("The switch point is the variable quantity", 0),
            ("Lock-in at a median 76% through the trial, with a p10–p90 "
             "spread of 28 percentage points — the solver's readiness is not "
             "synchronised to the operator's intent.", 1),
            ("Why this is the right candidate", 0),
            ("A policy conditioned on observation alone cannot infer how far "
             "through the hand-over it is: the same observation maps to "
             "different actions depending on an unobserved phase.", 1),
            ("Localised, not yet causally established — the fixed-timing arm tests it.", 0),
        ],
        cite=["State aliasing in the sense of Chen et al., Phase-Conditioned Imitation Learning, arXiv:2605.29407, 2026 (45% → 90%)."])

    # ---- 16. what we ruled out -----------------------------------------
    table_slide(
        prs, "Alternatives, and why each fails",
        kicker="Three plausible causes, each tested and refuted.",
        headers=["Hypothesis", "Test", "Result"],
        rows=[["Repaired finger labels drive it", "Train on the arm only",
               "*WORSE: 3.19× → 8.82×"],
              ["Not enough data", "Pool 16 → 63 episodes",
               "0.00174 → 0.00166"],
              ["Grip force too weak", "Squeeze against demand",
               "1210 N vs demos' 168 N"]],
        notes=["The finger-label hypothesis was mine. Arm-only training refuted it by more than doubling the gap — which also says the inconsistency lives in the arm, not the fingers.",
               "These objects need 0.23–3.6 N to hold statically. Grip force is not the binding constraint; reach is.",
               "Each refutation narrows the cause toward the hand-over."],
        col_w=[int(TITLE_W * 0.32), int(TITLE_W * 0.30), int(TITLE_W * 0.38)])

    # ---- 17. where this sits -------------------------------------------
    content(
        prs, "This result is not an outlier",
        kicker="Two 2026 papers find assistance-provenance effects that harm learning.",
        blocks=[
            ("BRIDGE: mixing action-label semantics is catastrophic", 0),
            ("NIST pulley 44.0% → 0%; battery insertion 10.0% → 0.0%, "
             "from mixing observed and desired labels.", 1),
            ("DATAFARM: task success does not imply learnable data", 0),
            ("“Raw TAMP trajectories provide surprisingly little benefit "
             "… despite successfully solving the target tasks.”", 1),
            ("Planner data 8.3% against human teleoperation 61.7%, attributed to "
             "behavioural distribution mismatch.", 1),
            ("The two literatures have not met", 0),
            ("Neither paper cites the shared-autonomy data-collection line, so "
             "that line has not absorbed this counterevidence.", 1),
            ("Our contribution: the comparison run in a pipeline where "
             "assistance is absent at deployment — no gating confound.", 1),
        ],
        cite=["BRIDGE, arXiv:2606.26603, 2026. DATAFARM, arXiv:2609.12316, 2026."])

    # ---- 18. what we can and cannot claim ------------------------------
    content(
        prs, "What is novel, and what is not",
        kicker="Three of these were claimed in the last twelve months.",
        blocks=[
            ("Already claimed — not ours", 0),
            ("FABCO: feasibility-certified collection, 3.2×, n = 15. They "
             "certify kinematic reproducibility; we certify wrench capacity.", 1),
            ("BlenDAgger: hand-over as a policy-relevant property — +30 "
             "points via 57% smoother transitions.", 1),
            ("LfDI: delayed interventions are “less effective at "
             "demonstrating error avoidance behaviors.”", 1),
            ("Absolute-vs-delta action spaces — established, including "
             "robomimic's 49–88% drops from added proprioception.", 1),
            ("Open, and defensible", 0),
            ("Hand-over VARIANCE — dispersion across a demonstration set, as "
             "distinct from bias, delay, or within-transition smoothness.", 1),
            ("Wrench-capacity certification as the mechanism.", 1),
            ("A clean decomposition of data quality from test-time architecture.", 1),
        ],
        cite=["FABCO, arXiv:2602.15351, 2026. BlenDAgger, arXiv:2609.37599, 2026. LfDI, arXiv:2610.17298, IROS 2026.",
              "Mandlekar et al., CoRL 2021, Lesson L4. Wen et al., Fighting Copycat Agents, NeurIPS 2020."])

    # ---- 19. metric validity -------------------------------------------
    content(
        prs, "A caveat about the metrics",
        kicker="The curation-metric literature is contested — in preprints that are not peer-reviewed.",
        blocks=[
            ("Smoothness may be a proxy for episode length", 0),
            ("Spectral arc length reaches AUROC 0.952 — but episode length "
             "ALONE reaches 0.957. Duration-matched, SAL falls to 0.628 while "
             "plain mean jerk rises to 0.896.", 1),
            ("Detection accuracy and policy benefit can decouple", 0),
            ("The best-detecting metric (AUROC 0.804) gave the WORST policy "
             "(13.3%); a 0.638 metric nearly matched an oracle.", 1),
            ("These audits are single-author preprints, NOT peer-reviewed — "
             "but the length confound is concrete, so we checked it.", 0),
            ("What this does to our result", 0),
            ("Our smoothness numbers survive duration matching (d = 0.80 → "
             "0.91), but the argument does not rest on them.", 1),
            ("The headline rests on action divergence — peer-reviewed, and "
             "the metric whose prediction the policy result confirmed.", 1),
        ],
        cite=["arXiv:2610.17427, 2026 (NeurIPS workshop poster, single author). arXiv:2606.10229, 2026 (preprint, single author).",
              "Belkhale, Cui & Sadigh, NeurIPS 2023 — peer-reviewed, and the source of our headline metric."])

    # ---- 20. methodological template -----------------------------------
    content(
        prs, "The design the next experiment copies",
        kicker="Separate the collection interface from the policy architecture, factorially.",
        blocks=[
            ("Wrench-ACT does this cleanly", 0),
            ("“Cross-condition ablations show that the bilateral data "
             "collection interface and the wrench action space each contribute "
             "independently to performance.”", 1),
            ("Precisely the decomposition HITL-TAMP's Appendix L left open, in an "
             "adjacent force-controlled setting.", 1),
            ("Applied here", 0),
            ("Interface factor: certified vs unassisted collection.", 1),
            ("Architecture factor: absolute vs delta action space; "
             "phase-conditioned vs observation-only policy.", 1),
            ("Crossing them separates “the data is better” from "
             "“the policy class handles this data better” — which "
             "no shared-autonomy data paper has done.", 1),
        ],
        cite=["Wrench-ACT, arXiv:2609.37552, 2026."])

    # ---- 21. two-week plan ---------------------------------------------
    content(
        prs, "Plan: the next two weeks",
        kicker="One decisive experiment, falsifiable, inside the window.",
        blocks=[
            ("Week 1 — collection", 0),
            ("Equal counts per condition and per object; ±5 cm spawn jitter; "
             "varied initial arm pose; GRASP-phase target, gains and live γ "
             "logged.", 1),
            ("Log the lock-in timestamp explicitly — it is currently "
             "reconstructed from events, which makes the central quantity an "
             "inference.", 1),
            ("Week 2 — the decisive experiment", 0),
            ("Three arms: baseline, contact-aware, and contact-aware with "
             "FIXED-TIMING lock-in.", 1),
            ("Prediction: if hand-over variance is the mechanism, arm 3 closes "
             "the learnability gap while keeping the task-success advantage. If "
             "it does not, the mechanism is something else.", 1),
            ("This is also the HITL-TAMP decomposition — assistance held "
             "constant, its timing varied.", 1),
            ("Deferred: ACT and implicit BC on current data; phase conditioning.", 0),
        ])

    # ---- 22. summary ---------------------------------------------------
    content(
        prs, "Summary",
        kicker="Three claims, in descending confidence.",
        blocks=[
            ("1.  Shared autonomy traded learnability for task success — measured", 0),
            ("Demonstrations 1.74–2.12× worse by action divergence; "
             "policies 3.6–3.9× worse by cross-validated loss; task "
             "success 94% vs 69%.", 1),
            ("The data metric predicted the policy result — which is what "
             "makes this more than an anecdote.", 1),
            ("2.  Hand-over variance is the likely exchange rate — localised", 0),
            ("28-point lock-in spread, consistent early and at the grasp. The "
             "fixed-timing arm tests it.", 1),
            ("3.  The field's consensus rests on three matched comparisons", 0),
            ("At least four systems assert the benefit untested; the closest "
             "paper has an unresolved confound; the counterevidence sits in a "
             "literature that does not cite this one.", 1),
        ])

    # ---- 23. thank you -------------------------------------------------
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = CMU_RED

    _, tf = _textbox(slide, 670560, 1700000, 10850880, 1200000)
    used = []
    _para(tf, used, "Thank you", 4400, bold=True, color=WHITE)

    _, tf = _textbox(slide, 670560, 3150000, 10850880, 2600000)
    used = []
    _para(tf, used,
          "Eleven months on this pipeline — from a grasp recommender that "
          "collided with its own object, to a certified-grasp teleoperation "
          "system with a measured answer about what its data is worth.",
          1700, color=WHITE)
    _para(tf, used,
          "Thank you to the AiPEX Lab for the run time, the hardware, and the "
          "arguments that changed my mind — several of which are on the "
          "slides above.",
          1700, color=WHITE, space_before=16)
    _para(tf, used,
          "The negative result is the one I am proudest of. It was the harder "
          "thing to measure, and it is the one worth passing on.",
          1700, color=WHITE, space_before=16)

    _, tf = _textbox(slide, 670560, 6050000, 10850880, 600000)
    used = []
    _para(tf, used, "Suresh Kumaar Jayaraman  ·  AiPEX Lab  ·  "
                    "Carnegie Mellon University", 1400, color=WHITE)

    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=Path("docs/IL_shared_autonomy.pptx"))
    args = ap.parse_args()
    p = build(args.out)
    print(f"-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
