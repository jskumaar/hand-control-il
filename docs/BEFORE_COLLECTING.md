# Changes to make before the next collection run

Written for: whoever runs the next teleop data-collection session, and whoever
edits `hand-control`'s logging.

Three of these are changes to **`hand-control`** (the collection side). One is a
decision about **what to collect**. Nothing in `handil` blocks collection — it
reads whatever the logs contain and the schema is versioned.

---

## 1. Log the GRASP-phase PD target (`hand-control`) — highest value

**The problem.** During the contact-aware GRASP phase the commanded torque is

```python
kp_eff, kd_eff = grasp_ctrl.effective_gains()
tau_ctrl[:N_ROBOT] = (kp_eff * (q_grasp_hold - data.qpos[:N_ROBOT])
                      + kd_eff * (...)) + tau_internal + tau_slip
```

`q_grasp_hold` is the real position command and **it is never logged**. Neither
is `kp_eff`, which is a time-varying product of `squeeze_pd_scale` (0.25),
`transport_pd_scale`, and `gamma/gamma_ref`. Without both, the finger target is
not recoverable: inverting the torque as if it were pure PD gives targets up to
20 rad.

**The consequence today.** `handil` has to replace those finger labels with
`qpos[t+1]` ("follow where the finger already went"), which affects **46% of
contact-aware rows** and leaves a squeeze margin of **0.002 rad median**. A
policy trained on that can track a closing hand but never command one.

**The fix.** Add three fields to the existing `_pose_trace.sample(...)` call in
`kinova_leap_pick_place.py` (it already records `q_robot`, `control_phase`,
`norm_force` etc., so this is three more kwargs):

| field | value | why |
|---|---|---|
| `q_grasp_hold` | the live 23-vector, or NaN outside GRASP | the actual position command |
| `kp_eff` | `grasp_ctrl.effective_gains()[0]` | makes the PD term invertible |
| `gamma_live` | the solved squeeze magnitude | lets a policy predict grip force |

With those, the GRASP finger target becomes exactly recoverable and the repair
can be deleted. This is the single change that most improves the data.

---

## 2a. Vary the OBJECT position — yes, and it needs a code change

**Measured, not assumed.** Object start positions in the current data have a
5–7 cm std per axis, which looks like useful variation until you split it:

| | x std | y std |
|---|---|---|
| **within** a run (repeated trials, same config) | **0.0000 m** | **0.0000 m** |
| between runs (session to session) | 0.057–0.075 m | 0.038–0.058 m |

So the spread is entirely **configuration drift across recording sessions**, not
deliberate variation. Within any single run every trial starts in exactly the
same place. That is the worst of both worlds: not controlled enough to be a clean
fixed-position dataset, and not varied enough to teach *where* to reach — a
policy can memorise one trajectory per object instead of learning the mapping
`obj_in_palm → reach direction`, which is the part that generalises.

**The code change.** `SEQ_SPAWN_XY` is hard-coded at
`kinova_leap_pick_place.py:1415`, and `args.no_randomize = True` is forced at
:459, so there is currently **no way to vary it from the CLI**. Add a flag:

```python
# argparse
_arg_parser.add_argument('--spawn-jitter', type=float, default=0.0,
                         help='metres of uniform XY jitter on the sequential '
                              'spawn point, resampled per trial')

# where the object is seated (near :1469, _spawn_object_at)
if args.spawn_jitter > 0:
    xy = [xy[0] + _rng.uniform(-args.spawn_jitter, args.spawn_jitter),
          xy[1] + _rng.uniform(-args.spawn_jitter, args.spawn_jitter)]
```

Use `--spawn-jitter 0.05` (±5 cm), which matches the between-run spread already
present and keeps every object inside the reachable envelope the layout comment
documents (x ∈ [−0.55, 0.55], y ∈ [−0.075, 0.575], 0.30–0.42 m from the base).
Drive it from `--seed` so a session is reproducible.

Also worth varying the object **yaw** (`quat` in `scene_objects.json` is already
per-entry), since grasp geometry depends on orientation — especially for the mug,
whose handle makes a canonical grasp ill-defined.

---

## 2b. Vary the initial ARM pose (`hand-control`) — do it, but not for the reason I first gave

Every existing episode starts from a **bit-identical** home pose (zero spread
across all 7 arm joints). That is unusual for an IL dataset and worth fixing on
general grounds — a policy that has only seen one start cannot be evaluated for
robustness to any other.

**Honest caveat:** I tested whether this is what blocks closed-loop success, and
it is not. Jittering the start at evaluation time (0 → 0.20 rad) left success at
0% and changed nothing, because there was no competence at jitter = 0 to lose.
So collect varied starts because it makes the dataset better, not because it is
expected to fix the current failure.

`PickPlaceEnv.reset(arm_jitter=..., object_jitter=...)` already exists on the
`handil` side for evaluation.

---

## 3. Check the `baseline+dexpilot` recordings (`hand-control`)

Those demos show a median **peak finger force of 2299 N** against 50 N for
`baseline+anyteleop`. That is not physical for a LEAP hand and indicates deep
interpenetration in the recordings. Worth reproducing one and watching it before
that cell is used for anything.

---

## 4. How many episodes

**Fix the metric before sizing the collection.** A single held-out split with
`val_frac=0.1` holds out only **1–2 episodes**, so the loss largely reports
*which* episode that was. Measured across episode counts (8 seeds each):

| episodes | mean val | sd | sd/mean |
|---|---|---|---|
| 6 | 0.297 | 0.192 | 65% |
| 10 | 0.119 | 0.153 | **128%** |
| 16 | 0.128 | 0.091 | 71% |
| 23 | 0.032 | 0.050 | **154%** |

Non-monotonic, and at 23 episodes the sd *exceeds* the mean — so **more episodes
do not stabilise a single-split metric**. (The *mean* does improve cleanly,
0.297 → 0.032, so more data genuinely helps; the metric just could not show it.)

`dataset.episode_folds()` does K-fold over episodes instead, holding out every
episode exactly once. That makes the metric **12× more stable** — seed-level sd
falls from 71% of the mean to 8%:

| | seed sd | as % of mean |
|---|---|---|
| single split, 16 eps | 0.0907 | 71% |
| **8-fold CV, 16 eps** | **0.0074** | **8%** |

On the stable metric, 16 episodes/cell already gives a readable comparison:
baseline 0.09467 ± 0.00277 vs contact_aware 0.09150 ± 0.01008, i.e. contact-aware
ahead by **3.3%**, Cohen d = 0.43. Note fold-level sd stays ~0.055 — episodes
genuinely differ a great deal — but CV averages that rather than sampling it.

**Recommendation.** Use K-fold CV for every reported number. Then, if the effect
is really ~3%, more episodes are worth collecting to lift it clear of the
remaining noise — but size that from a CV-based power calculation, not from the
single-split numbers, which were measuring the wrong thing. 40–50 per cell is a
reasonable target; equal counts per cell and per object, since the mug behaves
differently from the other three.

---

## What does NOT need changing

- **Phase labels.** `hand-control` already logs `phase_enter/PICK` correctly for
  every mode via `PhysicalPickTrigger` (`--physical-pick`, on by default). The
  gap was in `handil` reading the trace's `phase` column instead of the event
  stream; fixed.
- **`handil`'s extraction.** Schema is versioned (`SCHEMA_VERSION`), the manifest
  records `action_space` and `obs_extras`, and the loader refuses mismatched
  shards. New logs extract with no code change.
- **Action recovery.** Exact for everything outside the contact-aware GRASP
  phase, validated to 0.003 rad over 1500 steps.

---

## Before trusting any new result

Two of my own bugs invalidated results in the previous round. Re-check both on
new data:

1. **Per-episode validation split** (`dataset.episode_split`) — a per-sample
   split put near-duplicate frames on both sides and made losses look 25–85×
   better than they were.
2. **Rollout horizon ≥ 2500 steps** — at 400 steps every policy scored 0%; the
   chunked policy's first real pick appears at 2500. It plateaus there (4000 and
   6000 are identical), so 2500 is sufficient.
