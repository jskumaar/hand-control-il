# Collection commands for both conditions

Two conditions, retargeter held fixed so **pipeline is the only variable**.
`anyteleop` is the right retargeter: it is the only one with usable counts in
both arms today (16 baseline / 23 contact-aware).

## Prerequisites (one code change)

`--spawn-jitter` does not exist yet — see `BEFORE_COLLECTING.md` §2a. Without it
every trial in a run starts at exactly the same object position
(measured within-run std: 0.0000 m), which is the main thing worth fixing.
Also add the three GRASP log fields from §1.

## Condition A — baseline (raw teleop)

```bash
./start_teleop.sh sim anyteleop \
    --trial-log baselineA_$(date +%Y%m%d) \
    --sequential-spawn --order 1 \
    --spawn-jitter 0.05 \
    --seed 1 \
    --record on
```

## Condition B — contact-aware

```bash
./start_teleop.sh sim contact_aware_w_anyteleop \
    --trial-log contactB_$(date +%Y%m%d) \
    --sequential-spawn --order 1 \
    --spawn-jitter 0.05 \
    --seed 1 \
    --record on
```

## How many, and how to interleave

Target **40–50 successful episodes per condition** (today: 16 / 23).

Each run with `--sequential-spawn --order N` presents all four objects once, so
one run ≈ 4 episodes. That is ~11–13 runs per condition.

**Interleave the conditions** — alternate A, B, A, B rather than recording all
of A then all of B. Operator skill drifts over a session, and the current data
cannot distinguish that from a pipeline effect. Vary `--order 1..4` across runs
so object presentation order is counterbalanced, and use a fresh `--seed` per
run so the jitter differs.

Keep counts equal per object: the mug behaves differently from the other three
(43 collision hulls and a handle), so an imbalance there shows up as a
pipeline difference.

## Then analyse

```bash
# extract (handil needs no change; schema is versioned)
cd ../hand-control-il
uv run python scripts/extract_data.py \
    --logs ../hand-control/logs --out data/v3 --outcomes success

# the comparison, with every methodological fix applied
uv run python scripts/run_experiment.py --data data/v3 --seeds 12 --rollouts 4
```

`run_experiment.py` uses K-fold CV over episodes (not a single split), matched
episode counts, a fixed retargeter, and ≥2500-step rollouts — the four things
that each produced a wrong answer earlier in this project.
