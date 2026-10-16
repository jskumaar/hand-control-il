# Data collection: exactly what to run

Two conditions, retargeter held **fixed** (`anyteleop`) so pipeline is the only
variable. Verified: `start_teleop.sh` forwards `"$@"` to the python entry point,
and both new flags parse in the real script.

## 0. One-time setup

```bash
cd ~/sureshkj/hand-control

# the anyteleop retargeter needs the extra (plain `uv sync` prunes it)
uv sync --extra anyteleop

# the two patches from this work are in the working tree already; if you are on a
# clean checkout instead:
#   git apply ../hand-control-il/docs/hand_control_logging.patch
```

Then start the headset publisher in its own terminal:

```bash
./start_teleop.sh hands
```

## 1. The collection loop

**Interleave the conditions** — alternate A, B, A, B. Operator skill drifts over
a session and the existing data cannot separate that from a pipeline effect.
Bump `--seed` and `--order` together each pair.

```bash
cd ~/sureshkj/hand-control

# --- run pair 1 ---
./start_teleop.sh sim anyteleop --scene robocasa \
    --trial-log baseline_r01 --sequential-spawn --order 1 \
    --spawn-jitter 0.05 --seed 1 --record on

./start_teleop.sh sim contact_aware_w_anyteleop --scene robocasa \
    --trial-log contact_r01 --sequential-spawn --order 1 \
    --spawn-jitter 0.05 --seed 1 --record on

# --- run pair 2:  --order 2  --seed 2,  logs baseline_r02 / contact_r02
# --- run pair 3:  --order 3  --seed 3
# --- run pair 4:  --order 4  --seed 4
# --- then cycle --order 1..4 again with --seed 5,6,7,...
```

Each run presents all four objects once (~4 episodes), so **11–13 pairs** gets
both conditions from today's 16 / 23 to the 40–50 target.

`--seed` drives the per-run spawn offset, so the same seed reproduces the same
layout. The console prints it, e.g.

```
[spawn-jitter] this run presents objects at [-0.199, +0.495] (nominal [-0.2, 0.45], jitter ±0.050 m, seed=1)
```

Keep **counts equal per object**: the mug has 43 collision hulls and a handle, so
an imbalance there reads as a pipeline difference.

### What each flag is for

| flag | why |
|---|---|
| `--sequential-spawn` | one object at a time from a single point, so initial conditions are comparable |
| `--order 1..4` | cyclic-Latin presentation order, counterbalanced across runs |
| `--spawn-jitter 0.05` | **new** — per-run object position variation (within-run was 0.0000 m) |
| `--seed N` | reproducible jitter; vary across runs |
| `--record on` | overview + wrist MP4, for diagnosing a bad episode later |
| `--trial-log NAME` | required by `--sequential-spawn`; names the log directory |
| `--scene robocasa` | **required** — the default `pick_place` scene has only ONE object, so sequential spawn presents it and then has nothing to advance to. All ICRA runs used `robocasa` (4 objects). |

### Confirm on startup

The first lines must say:

```
[scene] spawning 4 YCB object(s) ... [robocasa]: obj_036_wood_block, obj_017_orange, obj_025_mug, obj_009_gelatin_box
[sequential-spawn] presenting 4 object(s) one at a time at xy=[...]; 3 stowed.
[spawn-jitter] this run presents objects at [...]            <- only if the patch is applied
```

If it says `spawning 1 YCB object(s) ... [pick_place]` you forgot `--scene robocasa`
and only one object will ever appear. If the `[spawn-jitter]` line is missing, the
patch in `hand_control_logging.patch` is not in the tree.

`[trial-log] enabled -> logs/NAME/ (mode=dexpilot)` is **cosmetic** for
`anyteleop` — the retargeter is fine (look for `[AnyTeleop] finger retargeting
via dex-retargeting`). That label is the lossy method field `handil` already
works around by reading the directory name.

## 2. Analyse

```bash
cd ~/sureshkj/hand-control-il

uv run python scripts/extract_data.py \
    --logs ../hand-control/logs --out data/v3 --outcomes success

uv run python scripts/run_experiment.py --data data/v3 --seeds 12 --rollouts 4
```

`extract_data.py` needs no change: the schema is versioned, and it automatically
uses the new `q_grasp_hold` / `kp_eff` fields when present (`action_source`
reports `logged_grasp_hold` instead of `inverted_torque`).

`run_experiment.py` applies all four methodological fixes this project needed —
K-fold CV over episodes, episode-level splits, matched episode counts, and
≥2500-step rollouts.

## 3. Sanity checks after the first pair

Before committing to 11 more pairs, confirm the new fields landed:

```bash
cd ~/sureshkj/hand-control-il
uv run python -c "
import numpy as np, glob
p = sorted(glob.glob('../hand-control/logs/contact_r01/pose_trace.npz'))[0]
d = np.load(p, allow_pickle=True)
print('has q_grasp_hold:', 'q_grasp_hold' in d.files)
print('has kp_eff      :', 'kp_eff' in d.files)
print('GRASP rows with a finite hold:',
      int(np.isfinite(d['q_grasp_hold']).all(1).sum()), '/', len(d['t']))
"
```

A nonzero count means the contact-aware finger labels are now exact and the
`qpos[t+1]` repair (46% of rows today) no longer applies.
