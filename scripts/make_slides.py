#!/usr/bin/env python3
"""Generate the IL-results slide deck, matching an existing deck's format.

Appends slides to a copy of the template so its theme, master and layout are
preserved exactly (the template is a Google-Slides export; rebuilding the theme
by hand would not match). Uses the template's own layout and the same bullet
hierarchy its slides use.

    python scripts/make_slides.py --template <deck.pptx> --out il_update.pptx
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from pptx import Presentation
from pptx.util import Pt

# (title, [(level, text), ...]) -- level 0/1/2 mirrors the template's bullets.
SLIDES: list[tuple[str, list[tuple[int, str]]]] = [
    ("Learning from teleop demos: setup", [
        (0, "Goal: does the collection method change what a policy can learn?"),
        (1, "Separate repo (handil), reads hand-control logs/ + models/ read-only"),
        (0, "Data: logs/ICRA_* — RoboCasa counter, 4 YCB objects, sequential spawn"),
        (1, "93 episodes after filtering: nominal-only, success-only, 4-object scenes"),
        (1, "contact_aware+vwj 38 · contact_aware+anyteleop 23 · baseline+anyteleop 16 · baseline+vwj 13"),
        (0, "Observation 78-d (state only): q/qd (46) + object-in-palm (7) + palm pose (7)"),
        (1, "+ contact forces (8) + object identity/size/mass (8) + lift/touch (2)"),
        (1, "All 23 robot joints are revolute: q in rad, qd in rad/s, action = target angle"),
        (0, "Action 23-d: target joint position (rad), or commanded torque (N·m)"),
        (0, "Algorithms: BC-MLP, BC-RNN, ACT (chunked CVAE), IBC — behind one registry"),
        (0, "Eval: closed-loop rollout in MuJoCo, scored with trial_logger.py's own criteria"),
    ]),
    ("Making the data learnable: 3 problems", [
        (0, "Actions are not logged — hand-control never records _dp_target"),
        (1, "Recovered exactly: target = qpos + (qfrc_applied − qfrc_bias) / Kp"),
        (1, "Exact because the arm Kd term is identically zero by construction"),
        (1, "Gains analytic, never fitted: diag(M)·30² at the home pose"),
        (1, "Validated: open-loop replay drifts 0.003 rad median over 1500 steps"),
        (0, "Method labels are lossy — events.jsonl collapses contact_aware_w_* variants"),
        (1, "Decomposed: pipeline from the event, retargeter from the directory name"),
        (0, "obj_in_palm is all-NaN in 41 of 44 episodes (logged only for the proximate object)"),
        (1, "Recomputed from palm + object poses; agrees with source to 1e-5 m over 11k rows"),
        (0, "Contact-aware GRASP phase is NOT invertible — internal-force torques are not PD"),
        (1, "kp_eff is a product of squeeze/transport scales × gamma/gamma_ref, none logged"),
        (0, "pose_trace.npz's phase column omits PICK for every baseline mode"),
        (1, "events.jsonl HAS it (4 phase_enter/PICK per run) — the source logs it correctly"),
        (1, "Fixed by reconstructing per-row phase from the event stream: now 93/93 episodes"),
        (1, "share APPROACH → PICK → TRANSPORT → PLACE, so no pipeline confound remains"),
    ]),
    ("Four bugs found (and what each cost)", [
        (0, "Continuous-joint clamp — source sets jnt_range = ±π on joints 0/2/4/6"),
        (1, "Only to bound IK sampling; enforcing it as physical cost 9 cm of palm error"),
        (1, "Demos drive joint 2 to 3.7–4.1 rad. Replay success 5% → 30%"),
        (0, "Object-index frame mismatch — obj_idx is in the TRACE's spawn order"),
        (1, "set_active_object indexes the SCENE's; they differ on 3 of 4 objects"),
        (1, "3 of 4 objects were replayed against the wrong object. Best cell 31% → 50%"),
        (0, "Silent CPU fallback — uv pulled cu130 torch against a 570 driver"),
        (1, "cuda.is_available() False → CPU, no error. Pinning cu128: 57× (76 min → 80 s)"),
        (0, "GRASP-label corruption — inverting non-PD torques gave 20 rad finger targets"),
        (1, "Repaired from qpos[t+1]; affects 47–59% of contact-aware rows"),
        (0, "Guards added: schema_version in manifest, obs_dim validated on checkpoint load"),
    ]),
    ("Result: the hypothesis is NOT supported (retraction)", [
        (0, "Earlier in this work I reported contact-aware as 2–3x more learnable."),
        (1, "That was wrong. Two bugs of mine: a validation split over SAMPLES not"),
        (1, "episodes (windows ~10 steps apart landed on both sides), and 3 seeds"),
        (1, "against an effect needing 24."),
        (0, "Properly measured — per-episode split, BC-MLP, 24 seeds per arm:"),
        (1, "baseline        0.0991 ± 0.0465"),
        (1, "contact_aware   0.1080 ± 0.0790"),
        (1, "effect −0.0089, Cohen d −0.14, Welch t −0.48, p = 0.63  → NULL"),
        (0, "Bootstrapping 3-seed draws from the 24: 44% make contact-aware look"),
        (1, "better by chance; my original +0.037 sits at the 82nd percentile"),
        (0, "BC-RNN under the same split is also a tie (0.1434 vs 0.1408, d = 0.04)"),
        (0, "The per-episode split also raised ALL losses 25–85x — the old numbers"),
        (1, "were measuring near-duplicate frames, not generalisation"),
    ]),
    ("Why: the consistency mechanism, measured", [
        (0, "Cross-episode spread of hand posture at peak grasp (lower = more canonical):"),
        (1, "gelatin box  0.407 → 0.112  (−72%)     object-in-palm spread −55%"),
        (1, "wood block   0.320 → 0.252  (−21%)     object-in-palm spread −46%"),
        (1, "orange       0.341 → 0.268  (−21%)     object-in-palm spread −46%"),
        (1, "mug          0.279 → 0.333  (+19%)     object-in-palm spread +7%"),
        (0, "The recommender imposes a repeatable grasp on 3 of 4 objects"),
        (1, "Mug is the exception — 43 collision hulls and a handle; no canonical grasp"),
        (0, "Caveat: contact-aware actions are intrinsically smoother (step p95 0.0095 vs 0.0147)"),
        (1, "So part of the val-loss win is structural — necessary evidence, not sufficient"),
        (0, "Stronger caveat: corr(action[t], qpos[t]) = 0.91, |action − qpos| = 0.016 rad median"),
        (1, "The action is nearly the observation, so a regressor scores well by near-copying"),
        (1, "=> validation loss is a WEAK proxy for task competence in this setup"),
    ]),
    ("Closed-loop: 0% success, and why", [
        (0, "Every trained policy fails the task. The cause is covariate shift, not a bug"),
        (0, "The policy is a good predictor and a bad controller:"),
        (1, "Fed the DEMO's observations it predicts the demo action to 0.007 rad median"),
        (1, "Acting on its OWN observations: 0.088 rad off after 1 step, 1.12 rad by step 50"),
        (1, "Ends commanding 3.67 rad from its start, action steps 2× the demo's"),
        (0, "Mechanism: data is dense ALONG trajectories, 50× sparser BETWEEN them"),
        (1, "Along-trajectory spacing 0.006 rad; nearest other-episode state 0.31 rad"),
        (1, "No demo says 'you are 0.09 rad off, here is how to recover' — they never were"),
        (0, "The plumbing is proven sound, which is what localises the failure to learning:"),
        (1, "Env faithful — a recorded human trajectory succeeds in it (50% on the best cell)"),
        (1, "Labels correct — 0.003 rad replay drift;  policy fits — 0.007 rad teacher-forced"),
    ]),
    ("Arm vs hand: two different failures", [
        (0, "Matched-state analysis (leave-one-episode-out kNN, 1 neighbour per episode):"),
        (0, "ARM = coverage-limited"),
        (1, "Action disagreement → 0.002–0.008 rad as observations coincide"),
        (1, "Correlation with observation distance 0.85–0.90"),
        (1, "Demos AGREE about the arm; they just do not co-visit states → fixable with data"),
        (0, "HAND = unidentifiable target"),
        (1, "Disagreement PLATEAUS at 0.035–0.073 rad at near-zero observation distance"),
        (1, "Correlation with observation distance only 0.12–0.52"),
        (1, "Two demos in the same state command finger targets differing by ~0.05 rad"),
        (0, "Tested fix: add object identity to the observation → REFUTED"),
        (1, "Baseline hand floor 0.0729 rad WITH identity vs 0.0584 without — no improvement"),
        (1, "Cause of the hand floor is still unknown"),
    ]),
    ("What else was tried", [
        (0, "Torque action space — right LABEL, wrong COMMAND"),
        (1, "Only faithful label for the GRASP phase (gains unlogged); needs no inversion"),
        (1, "But 4× worse drift closed-loop (17.7 vs 4.19 rad @200)"),
        (1, "Position commands self-correct; torque integrates error twice"),
        (0, "Internal-force squeeze ported from GraspController — helps a trained policy"),
        (1, "Halves drift (2.26 → 1.12 rad @50) and closes 14 cm of the reach gap"),
        (1, "Imports the source controller rather than reimplementing the null-space solve"),
        (0, "ACT action chunking — 6× lower early drift (0.183 vs 1.595 rad @50)"),
        (0, "No effect: velocity injection · achieved-position servoing · 4× interpolation"),
        (0, "No effect: pooling 16 → 63 episodes (0.00174 → 0.00166) — same start state"),
        (0, "Not the limit: grip force. Squeeze reaches 1210 N; statics need 0.23–3.6 N"),
    ]),
    ("Current status", [
        (0, "Infrastructure working and validated — 15 commits, 71 tests passing"),
        (1, "Extraction, 4 algorithms, rollout env + squeeze, replay calibration, MP4 recording"),
        (1, "GPU training ~5 s/cell; full 3×4 matrix in 3.9 min"),
        (0, "Replay ceiling (demonstrations driven through the env): the target to beat"),
        (1, "baseline+anyteleop 50% pick / 50% success  ·  overall 16% / 11% (n=93)"),
        (0, "Replay cannot adjudicate contact-aware — success tracks GRASP-repair fraction"),
        (1, "0% repair cells reach 15–50%; 47–59% repair cells reach 0%"),
        (1, "Our label reconstruction, not the data, is what replay cannot execute"),
        (0, "Headline: hypothesis supported OFFLINE (2.1–3.1× BC-MLP, mechanism measured)"),
        (1, "No policy performs the task closed-loop — that remains open"),
    ]),
    ("Next steps", [
        (0, "1. Recovery data (DART-style noise injection) — highest expected value"),
        (1, "Perturb demo states, relabel with the P-action returning to the demo path"),
        (1, "Arm labels are known-consistent, so relabelling is sound; no new teleop needed"),
        (1, "Directly attacks the 50× along-vs-between sparsity"),
        (0, "2. ACT on the v2 observation — already implemented, 6× lower early drift"),
        (1, "Chunking gives 32× fewer compounding opportunities per prediction"),
        (0, "3. Chase the hand floor — object identity was NOT the missing variable"),
        (1, "Leading hypothesis: grasp-contact geometry (which finger, which surface)"),
        (1, "The recommender's rec_local carries it; not logged, maybe reconstructible"),
        (1, "Fix two measurement artifacts first: 13.3% frozen frames; one-hot z-scoring"),
        (0, "4. Alternative: accept the floor, give the hand a distributional head (ACT CVAE)"),
        (0, "5. New demos with jittered starts — principled fix for the arm; needs the rig"),
    ]),
    ("Open risks / caveats", [
        (0, "Statistical power: key cells are 16 vs 23 episodes after nominal-only filtering"),
        (1, "BC-RNN contact-aware advantage already vanishes at this n"),
        (0, "Part of the offline win is structural (smoother contact-aware actions)"),
        (0, "Hand-floor diagnosis rests on a metric with two known defects (being fixed)"),
        (0, "obj_mass is constant per object at rollout — the env cannot reproduce the"),
        (1, "ablation sweep's scaled mass/friction, so those runs are excluded"),
        (0, "Earlier claim that PICK leaks the teleop mode was WRONG about the cause —"),
        (1, "it was our extractor reading the trace column, not hand-control's logging"),
        (0, "Data quality: baseline+dexpilot demos peak at 2299 N finger force"),
        (1, "Not physical for a LEAP hand — indicates interpenetration in those recordings"),
        (0, "Single scene, single start pose, 4 objects — no generalisation claim is testable yet"),
    ]),
]


def add_slide(prs, layout, title: str, bullets: list[tuple[int, str]]):
    slide = prs.slides.add_slide(layout)
    slide.shapes.title.text = title

    body = None
    for ph in slide.placeholders:
        if ph.placeholder_format.idx != 0:
            body = ph
            break
    if body is None:
        return slide

    tf = body.text_frame
    tf.clear()
    for i, (lvl, text) in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text
        p.level = lvl
        # The template's body runs 18pt at lvl 0; step down so dense slides fit.
        size = 14 if lvl == 0 else 12
        for run in p.runs:
            run.font.size = Pt(size)
    return slide


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("il_update.pptx"))
    ap.add_argument("--keep-template-slides", action="store_true",
                    help="keep the template's own slides ahead of the new ones")
    args = ap.parse_args()

    prs = Presentation(str(args.template))
    # Reuse the layout the template's own content slides use, so theme,
    # placeholder geometry and bullet styling all match.
    layout = prs.slides[-1].slide_layout

    if not args.keep_template_slides:
        xml_slides = prs.slides._sldIdLst
        for sld in list(xml_slides):
            rId = sld.get(
                "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
            prs.part.drop_rel(rId)
            xml_slides.remove(sld)

    for title, bullets in SLIDES:
        add_slide(prs, layout, title, bullets)

    prs.save(str(args.out))
    print(f"{len(SLIDES)} slides -> {args.out}")
    for i, (t, b) in enumerate(SLIDES, 1):
        print(f"  {i:2d}. {t}  ({len(b)} bullets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
