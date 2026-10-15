# Collection commands for both conditions

Two conditions, retargeter held fixed so **pipeline is the only variable**.
`anyteleop` is the right retargeter: it is the only one with usable counts in
both arms today (16 baseline / 23 contact-aware).

## Prerequisites

`--spawn-jitter` is **implemented** in the working tree of `hand-control`
(`kinova_leap_pick_place.py`, two hunks). It is left uncommitted because that
file already carried ~560 lines of your unrelated in-progress work when this was
written, and those are not mine to commit. Review and commit alongside your own,
or apply `docs/spawn_jitter.patch` to a clean tree.

It applies the offset **once per run**, drawn from `--seed`: within a run every
trial still presents its object at the same place (the four objects already
differ, so within-run variation is not what the dataset lacks), while different
runs differ. Verified: 12 seeds give a 0.030 m std per axis — comparable to the
0.038–0.075 m drift already present between runs, but deliberate and
reproducible — and the clamp keeps every point inside the envelope 93 real
successful demos actually occupy (x ∈ [−0.36, −0.12], y ∈ [0.34, 0.56],
0.59–0.68 m from `base_link`).

> The layout comment in `scene_objects.json` says objects should sit 0.30–0.42 m
> from the base. That is **stale** for the robocasa counter + sequential spawner:
> real successful demos sit at 0.61–0.76 m. The clamp follows the measurement,
> not the comment.

Still outstanding: the three GRASP log fields from `BEFORE_COLLECTING.md` §1,
which are what make the contact-aware finger labels recoverable.

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
