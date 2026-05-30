# DREAM Social-Friendliness Pipeline

**End-to-end: dataset extraction → behaviour analysis → social-friendliness scoring → RL training in the simulator.**

This document is the architectural map for the v5 system.  Read it
top-to-bottom to understand which artifact is produced where, what
each metric measures, and how the offline analysis feeds the
online RL design.  Each stage links to the script and key file:line
that implements it.

---

## Stage 0 — Inputs

The pipeline ingests one or more naturalistic trajectory recordings
through [`tracks_import.py`](../../tracks_import.py:1):

| Family       | Source                               | Special handling             |
| ------------ | ------------------------------------ | ---------------------------- |
| highD / inD / rounD / uniD / exiD | drone-tool `*_tracks.csv` triplets | `read_from_csv`              |
| SQM-N-4      | Excel (`frenet-up.xlsx`, `frenet-down.xlsx`) + road map | `_build_sqm_dataset`         |
| YTDJ-3       | Single CSV (`frenet.csv`) + road map | `_build_ytdj_dataset`        |
| XAM-N-5/6    | `pixel.csv` + `frenet.csv`           | `_build_xam_dataset`         |

All five formats are normalised into one schema (per-track dicts of
arrays: `xCenter`, `yCenter`, `xVelocity`, `yVelocity`, `heading`,
`laneletId`, `laneChange`, `lonAcceleration`, etc.).

---

## Stage 1 — Feature extraction (schema v5)

Entry point: `python -m rl.data.historical_extractor ... --include-social`

[`historical_extractor.py:493`](historical_extractor.py:493) walks
every car/van/motorcycle track that satisfies
`MIN_TRACK_DURATION_S=4 s` and `MIN_EGO_SPEED=2 m/s`, and emits one
record per timestep.

### 1A. Per-frame ego features (v3 baseline)

Computed by `_per_frame_features` ([historical_extractor.py:309](historical_extractor.py:309)):

* slotted neighbours `(front/rear) × (same/left/right)`,
  `LANE_WIDTH_ASSUMED=3.5 m`, `PERCEPTION_RANGE=60 m`;
* analytic risk proxy (`risk_at`, `risk_corridor_tau`,
  `risk_gradient`) calibrated to `DRIFTInterface.get_risk_cartesian`;
* lane-wise utility `U_j = gap_score − dv_score − risk_score`,
  identical to `rl/reward/reward_fn.py::_lane_advantage` so that
  offline mining and on-line reward use the same scoring;
* primary action label `action_9way` and the decomposed
  `lane_delta_label`, `speed_mode_label`.

### 1B. Outcome labels (v3, 3-s look-ahead)

`future_lane_delta`, `future_speed_gain`, `future_gap_gain`,
`future_risk_change`, `lane_change_success`, `near_miss_future`,
`collision_future`, `blocked_by_leader_flag`, `escape_success_flag`,
`lane_change_advantage_flag`, `short_horizon_return_proxy`.

Thresholds: `NEAR_MISS=6 m`, `COLLISION=2 m`, `BLOCKED=25 m & 0.7×`,
`TAU_D=5 m`, `TAU_R=0.2`.

### 1C. Per-(ego, frame) social features (v4)

Computed by helpers in
[`social_features.py`](social_features.py:1):

* **Courtesy** (`courtesy_block`): rear-vehicle outcome via
  `_follower_trajectory` look-up — `rear_decel_peak_3s`,
  `rear_ttc_now/_after/_delta`, `rear_thw_now/_after/_delta`,
  `hard_brake_imposed_flag`, `bad_cut_in_flag`.
* **Decision quality** (`decision_block`):
  `missed_opportunity_flag` (`best_adv > 0.6` ∧ blocked ∧ keep),
  `bad_lane_change_flag` (LC with `best_adv < −0.3`).
* **Per-ego BEV field externality** (`RiskFieldQuery.field_grid` →
  `field_metrics`): `risk_mass_total`, `risk_mass_others`,
  `risk_gradient_peak`, `risk_flux_backward`, `risk_field_entropy`.
* **Composite scores** (`composite_scores`): `safety_score`,
  `progress_score`, `courtesy_score`, `social_friendliness_score`,
  and the 5-class `social_class ∈ {good, defensive, aggressive,
  passive, harmful}`.

### 1D. Frame-level traffic-efficiency features (v5, NEW)

A pre-pass over `sorted(frame_index.keys())`
([historical_extractor.py:559](historical_extractor.py:559)) produces
one frame-aggregate dict, broadcast onto every ego row in that
frame:

| Metric                                | Formula                                                     |
| ------------------------------------- | ----------------------------------------------------------- |
| `num_agents_frame`                    | `\|frame_index[t]\|`                                          |
| `close_pair_count`                    | unique pairs with `d < 30 m`                                |
| `closing_pair_count`                  | close pairs with positive closing rate                      |
| `interaction_density`                 | `close_pair_count / N_t`                                    |
| `risk_mass_frame`                     | `Σ_pairs exp(−d²/2σ²)·(1+max(0,closing)/V0)`                |
| `risk_mass_per_agent`                 | `risk_mass_frame / N_t`                                     |
| `risk_per_close_pair`                 | `risk_mass_frame / close_pair_count`                        |
| `risk_flux_backward_frame`            | sum over closing pairs where j is downstream of i           |
| `backward_risk_flux_ratio`            | `risk_flux_backward_frame / risk_mass_frame`                |
| `mean_speed_frame`, `speed_variance_frame` | over all agents in frame                                    |
| `total_progress_rate_frame`           | `Σ \|v\|`                                                     |
| `risk_mass_delta_frame`               | `R(t) − R(t−1)`                                             |
| `risk_mass_growth_rate_frame`         | `delta / dt`                                                |
| `shockwave_onset_flag`                | growth_rate > 0.05 ∧ Δspeed_var > 0.5                       |
| `risk_adjusted_progress`              | `total_progress / (1 + risk_mass_per_agent)`                |
| `social_traffic_efficiency_index`     | weighted blend of progress, agent-normalised risk, backward flux, speed variance |

Constants in [`social_features.py:CLOSE_PAIR_DIST_M`,
`SHOCKWAVE_GROWTH_THR`, `STEI_W_*`].

> **PDE-overlay extension point.** `RiskFieldQuery(mode='pinn'|'drift',
> callable_=...)` accepts the trained PINN
> (`pinn_risk_field.FieldInterpolator.query`) or a precomputed
> numerical PDE snapshot bundle so the per-ego BEV metrics can be
> re-computed against the true propagated field instead of the
> analytic kernel — without touching the rest of the pipeline.

### 1E. Output

The extractor writes one `bc_<dataset>_v5.npz` with ~80 arrays per
sample plus a sidecar split manifest.  Schema sanity is in
`summarize_dataset` ([historical_extractor.py:1004](historical_extractor.py:1004)).

```bash
python -m rl.data.historical_extractor \
    --dataset-format highD --data-dir data/highD --recordings all \
    --include-social \
    --out-path rl/checkpoints/bc_highd_v5.npz
```

---

## Stage 2 — Behaviour analysis (three figures)

All three plotting scripts share the same `--ego-id <int>`
(case study) / `--list-egos` / `--from-dataset` interface.

### Figure 1 — Tactical choice ([`plot_behavior_summary.py`](plot_behavior_summary.py:1))

Answers: **Did the ego make tactically reasonable lane / speed
choices?**

| Panel | Reads                                                                |
| ----- | -------------------------------------------------------------------- |
| (a)   | 9-way action histogram                                               |
| (b)   | lane-delta and speed-mode marginals                                  |
| (c)   | outcome rates (LC success/adv \| LC, blocked, escape, near-miss)     |
| (d)   | `future_risk_change` violins by lane action                          |
| (e)   | calibration: P(LC) vs binned `best_adv`                              |
| (f)   | dataset-comparison heatmap                                           |

### Figure 2 — Social externality ([`plot_social_externality.py`](plot_social_externality.py:1))

Answers: **Did the ego's chosen action burden the rear / target-lane
follower, or the wider local field?**

| Panel | Reads                                                                       |
| ----- | --------------------------------------------------------------------------- |
| (a)   | `rear_decel_peak_3s` violins by lane action                                 |
| (b)   | bars: `hard_brake_imposed_flag`, `bad_cut_in_flag`, `rear_ttc<0`, `rear_thw<0` |
| (c)   | bars: `missed_opportunity_flag`, `bad_lane_change_flag`, `lc_advantage \| LC`, `escape \| blocked` |
| (d)   | scatter: `risk_mass_others` × `risk_gradient_peak`, coloured by `social_class` |
| (e)   | 5-class breakdown                                                           |
| (f)   | progress × courtesy Pareto, colour = safety                                 |

### Figure 3 — Traffic-efficiency externality ([`plot_traffic_efficiency.py`](plot_traffic_efficiency.py:1))

Answers: **Was the *whole frame* orderly, or was the scene producing
stop-and-go / shockwave behaviour?**

| Panel | Reads                                                                       |
| ----- | --------------------------------------------------------------------------- |
| (a)   | `risk_mass_per_agent` × `interaction_density`, coloured by STEI             |
| (b)   | `backward_risk_flux_ratio` violins by ego action                            |
| (c)   | `risk_adjusted_progress` median + IQR vs `num_agents_frame`                 |
| (d)   | STEI histogram                                                              |
| (e)   | `risk_mass_growth_rate_frame` time-series, shockwave onsets marked          |
| (f)   | recording-level summary table (mean N, mean per-agent risk, STEI, onset rate, total prog/total risk) |

### Per-agent vs aggregate

`--ego-id 27` filters every panel to one trajectory.  Aggregate
results bias toward egos with the most samples (long tracks); for
the paper, **report both**.

---

## Stage 3 — Social-friendliness scoring

Scoring lives at three granularities, all derived from v5 features.

### 3A. Per-(ego, frame) — `composite_scores` ([social_features.py:381](social_features.py:381))

```text
safety   = 1 − clip(max(0, future_risk_change)/0.6
                    + 0.6·near_miss + 1.0·collision)
progress = clip(max(0, future_speed_gain)/5.0
                + 0.5·escape_success − 0.5·missed_opportunity)
courtesy = 1 − clip(max(0,−rear_decel_peak)/6.0
                    + max(0,−rear_ttc_delta)/4.0
                    + 0.5·hard_brake_imposed + 0.5·bad_cut_in)

social_friendliness =
    0.40·safety + 0.30·progress + 0.30·courtesy
    − 0.10·missed_opp − 0.10·bad_lane_change
```

5-class label rule (precedence top-down): collision/hard_brake/bad_cut → harmful; missed_opportunity → passive; bad_lane_change → aggressive; safety∧courtesy high ∧ progress low → defensive; all high → good.

### 3B. Per-frame — Social Traffic Efficiency Index (STEI)

```text
STEI = 1.0·progress_rate
     − 0.30·risk_mass_per_agent
     − 0.30·risk_others_per_agent
     − 0.20·backward_risk_flux_ratio
     − 0.10·speed_variance_normalised
     − 0.10·hard_brake_rate
```

Defaults defined in `social_features.STEI_W_*` so the policy reward
can mirror them.

### 3C. Per-recording

Aggregate `STEI`, `shockwave_onset_rate`, `risk_recovery_time` (TODO,
backlog), and `social_class` distribution.  These are what the
panels in Figure 3 row 3 expose for cross-dataset comparison.

---

## Stage 4 — RL algorithm design

The offline analysis is meant to drive **two concrete changes** to
the on-line RL stack in `rl/`:

### 4A. Reward shaping (`rl/reward/reward_fn.py`)

Current per-step reward (v2) already covers progress, speed,
comfort, lane-keep, near-miss, CBF, lane-advantage, inaction,
commitment.  The v5 extension adds three new terms whose weights
are kept consistent with the offline scoring constants:

| Reward term     | Online expression                                                                  | Offline analogue           |
| --------------- | ----------------------------------------------------------------------------------- | -------------------------- |
| `r_courtesy`    | `−W_COURTESY · max(0, −rear_decel_step) / 6.0 − 0.5 · hard_brake_imposed_step`     | `courtesy_score`           |
| `r_externality` | `−W_EXT · backward_risk_flux_ratio_now`                                             | Figure 3 panel (b)         |
| `r_stei`        | `+W_STEI · STEI_step` (per-step variant; signal = scene's running average)         | Figure 3 panel (d)         |

`rear_decel_step` is computed from `Surrounding_model.py` follower
acceleration; `backward_risk_flux_ratio_now` reuses the
`RiskFieldQuery` machinery during the env step so analytic offline
and PINN/PDE online are interchangeable.

Tuning protocol:
1. extract one full dataset with `--include-social`;
2. for each candidate `(W_COURTESY, W_EXT, W_STEI)`, replay the
   trajectories through `compute_reward`;
3. compare the resulting reward distribution against the
   `social_friendliness_score` distribution from Figure 2 — choose
   weights that minimise the KL divergence on the per-bin histogram.

### 4B. Constraint specification (`rl/safety/`)

The CBF/MPC layer can be tightened with:

* **per-step courtesy budget** — abort actions for which the
  predicted `rear_decel_peak_3s < HARD_BRAKE_DECEL_MPS2`;
* **frame-level externality budget** — penalise the policy when
  `backward_risk_flux_ratio > 0.4` for more than 1 s (shockwave
  precursor);
* **STEI floor** — terminate the episode early with a strong
  penalty when STEI drops below the dataset's 5th percentile for >
  3 s (clear loss-of-flow scenario).

These three are *constraints*, not reward terms — they are intended
for constrained-RL (CPO/PPO-Lagrangian) variants in
[`rl/train_decision_ppo.py`](../train_decision_ppo.py:1).

### 4C. Curriculum and case-study sampling

Use the 5-class `social_class` label as a curriculum knob:

1. **Stage 0** — train on `social_good` ∪ `social_defensive` only
   (clean human imitation).
2. **Stage 1** — add `social_passive` so the policy learns to
   distinguish appropriate caution from missed opportunities.
3. **Stage 2** — add `social_aggressive` ∪ `social_harmful` as
   *negative* examples in a contrastive auxiliary head.

Per-agent slicing (`--ego-id`) is the ablation tool: pick three
representative ids — one good, one aggressive, one passive — and
verify the trained policy reproduces the good behaviour and avoids
the aggressive/passive ones in equivalent frames.

### 4D. Training loop

End-to-end shape:

```
                ┌──────────────┐
data/<format> → │ extractor v5 │ → bc_<ds>_v5.npz
                └──────┬───────┘
                       ▼
                ┌──────────────┐    ┌────────────────────────┐
                │ plot_*       │ →  │ Figures 1, 2, 3,       │
                │              │    │ console summaries      │
                └──────┬───────┘    └────────────────────────┘
                       ▼
              ┌────────────────────┐
              │ train_bc.py        │ — multi-head BC: action,
              │                    │   future_risk_change,
              │                    │   social_class
              └────────┬───────────┘
                       ▼
              ┌────────────────────┐    ┌────────────────────┐
              │ train_decision_ppo │ ←──│ DRIFT simulator    │
              │  (reward = task    │    │ (rl/env/...)       │
              │   + courtesy + ext │    │ exposes RiskField  │
              │   + STEI; CBF      │    │ live              │
              │   constraints)     │    └────────────────────┘
              └────────┬───────────┘
                       ▼
              ┌────────────────────┐
              │ eval.py rolls out  │
              │ → recompute v5     │
              │   metrics in env   │
              │ → compare to       │
              │   human dataset    │
              │   distributions    │
              └────────────────────┘
```

The closing eval step is what turns this into a research claim:
**we can show the trained policy reproduces the human population's
STEI, courtesy_score, and shockwave_onset_rate distributions** —
not merely matches collision/progress targets.

---

## Stage 5 — Reproducibility commands

```bash
# 1. Smoke (one recording, capped tracks, ~80 s)
python -m rl.data.historical_extractor \
    --dataset-format highD --data-dir data/highD --recordings 01 \
    --limit-tracks 30 --include-social \
    --out-path rl/checkpoints/bc_v5_smoke.npz --no-manifest

python -m rl.data.plot_behavior_summary  --inputs rl/checkpoints/bc_v5_smoke.npz --out figures/smoke_tactical
python -m rl.data.plot_social_externality --inputs rl/checkpoints/bc_v5_smoke.npz --out figures/smoke_social
python -m rl.data.plot_traffic_efficiency --inputs rl/checkpoints/bc_v5_smoke.npz --out figures/smoke_traffic

# 2. Per-agent case study (single ego id)
python -m rl.data.plot_traffic_efficiency \
    --inputs rl/checkpoints/bc_v5_smoke.npz --ego-id 27 \
    --out figures/smoke_traffic_ego27

# 3. Full extraction + comparison (multi-dataset heatmap activates)
python -m rl.data.historical_extractor --dataset-format highD --data-dir data/highD --recordings all \
    --include-social --out-path rl/checkpoints/bc_highd_v5.npz
python -m rl.data.historical_extractor --dataset-format exiD  --data-dir data/exiD  --recordings all \
    --include-social --out-path rl/checkpoints/bc_exid_v5.npz
python -m rl.data.plot_behavior_summary \
    --inputs rl/checkpoints/bc_highd_v5.npz rl/checkpoints/bc_exid_v5.npz \
    --labels highD exiD --out figures/dataset_compare
```

---

## Stage 6 — Open backlog (what is NOT yet implemented)

These are the deliberate gaps an expert reviewer should know about:

1. **Risk recovery time** (per-frame: time until `risk_mass_frame`
   returns below pre-event baseline).  Needs a stateful per-event
   tracker over frames; left as a follow-up.
2. **Yield-given / yield-received flags.**  Currently approximated
   by `r_inaction` in the reward but not labelled in the dataset.
   Needs a second-pass labeller that scans neighbour decelerations
   *induced* by ego pre-emption.
3. **Merge-cooperation features** (`gap_offered_to_merger`,
   `accepted_gap_size`, `late_merge_flag`, `zipper_score`).  The
   merge-detection itself is non-trivial; the pipeline emits the
   raw lane-id transitions, but the cooperation labels need a
   merger-detection helper.
4. **PINN / PDE override mode wired end-to-end.**
   `RiskFieldQuery(mode='pinn'/'drift')` accepts a callable; we
   ship an analytic mode but no glue for loading a trained
   `pinn_risk_field.pt` snapshot bundle on the extraction CLI.  A
   `--field-mode pinn --pinn-checkpoint pinn_risk_field.pt` flag is
   the next addition.
5. **STEI normalisation per dataset.**  Current weights produce
   STEI ≈ 28 on highD because progress dominates; the per-dataset
   normaliser (5th–95th percentile rescale) is recommended before
   cross-dataset comparison.
6. **Curriculum sampler** in `train_bc.py` that consumes the
   `social_class` column.  The class label is in the npz; the
   trainer is not yet using it.
7. **Per-(ego, frame) `risk_others_per_agent`** is currently
   recomputed online by `panel_risk_density`; an extracted column
   would simplify cross-dataset analysis.
