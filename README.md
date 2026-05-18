
## Risk-Informed Physics-Based Policy Learning for Externality-Aware Autonomous Driving

### This repo is for HKU IDS RPG programme DATA8007 Project Submission

Relevant preliminmary model used could be found at [DRIFT](https://github.com/PeterWANGHK/DRIFT.git)

![Methodology graph](assests/methodology_8007.jpg)


### PINN training
```
python pinn_risk_field.py --dataset inD --recording all --epochs 3000 --q_smooth --w_data 1.0 --w_phys 0.5 --w_ic 0.2 --w_bc 0.2 --w_smooth 0.3 --n_data 4096 --n_colloc 4096 --pts_per_snap 400 --save_model pinn_inD_all.pt
```
### demonstrations of the numerically solved risk field and PINN generated risk field:
![PINN_examples](assests/DRIFT_PINN_1.gif)
### PINN field output in various environment configurations (highway, highway with merging, roundabout, intersection)
![PINN_scenario](assests/pinn_result.jpg)

(The environment configurations are forked from [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv.git))

### Dataset processing
```
# Load the recorded trajectories:
python run_track_visualization.py --dataset [name of the dataset (e.g., highD; SQM-N-4)] --recording 00
# Example: load the behaviors from the SQM-N-4 dataset and store into .npz file 
python -m rl.data.historical_extractor --dataset SQM-N-4 --data-dir data/SQM-N-4   --out-path rl/checkpoints/bc_sqm_v3.npz
```
### RL training and evaluation in heterogeneous traffic (PPO only)
```
# 1. Extract ALL recordings into one dataset
python -m rl.data.historical_extractor --data-dir data/exiD --recordings all --out-path rl/checkpoints/bc_dataset_full.npz --horizon-sec 1.5

# 2. BC pretrain on the full dataset
python -m rl.train_bc --dataset rl/checkpoints/bc_dataset_full.npz --out rl/checkpoints/decision_policy_bc.pt

# 3. PPO fine-tune (with the new opportunity-aware reward)
python -m rl.train_decision_ppo --bc-checkpoint rl/checkpoints/decision_policy_bc.pt --out rl/checkpoints/decision_policy_ppo.pt --total-steps 200000

# 4. Evaluate (on both pure car traffic or heterogeneous traffic)
# in heterogenous traffic with truck-trailer occlusion and merging
python highway_test.py --models RL-PPO IDEAM DREAM --rl-decision-checkpoint rl/checkpoints/decision_policy_ppo.pt --steps 250
# in pure car traffic
python highway_test.py --scenario-mode purecar --ego-start-lane center --rl-policy-mode decision --rl-decision-checkpoint rl/checkpoints/decision_policy_ppo.pt --models all --mode single
# in suddent merging scenario: (compare against baseline MPC-CBF)
python uncertainty_merger.py --models "RL-PPO" "IDEAM" --steps 100 --rl-policy-mode ppo --rl-checkpoint rl/checkpoints/ppo_best.pt --save-dir figsave_merger_rl_vs_ideam --save-frames false

```
### Complete Implementation (updated on 25 Apr 2026)
```bash
# Train BC (if not already trained)
python -m rl.train\_bc --out rl/checkpoints/decision\_policy\_bc.pt

# Train PPO v3
python -m rl.train\_decision\_ppo \\
  --bc-checkpoint rl/checkpoints/decision\_policy\_bc.pt \\
  --out rl/checkpoints/decision\_policy\_ppo\_v3.pt \\
  --total-steps 200000 --rollout-steps 2048 \\
  --entropy-coef 0.05 --lr 1e-4 \\
  --log-path rl/logs/decision\_ppo\_v3\_log.json

# Main paper figure
python -m rl.plot\_training\_curves \\
  --logs rl/logs/decision\_ppo\_v3\_log.json \\
  --out figures/ppo\_training.pdf --diagnostic

# Evaluation — merger scenario
python uncertainty\_merger.py \\
  --rl-policy-mode decision \\
  --rl-decision-checkpoint rl/checkpoints/decision\_policy\_ppo\_v3.pt \\
  --steps 100 --models all --save-dir figsave\_merger\_v3\_rl

# Evaluation — 3-lane dense highway
python highway\_test.py \\
  --rl-policy-mode decision \\
  --rl-decision-checkpoint rl/checkpoints/decision\_policy\_ppo\_v3.pt \\
  --steps 400 --save-dir figsave\_test\_v3\_rl
```

## Datasets used in this project (download links):
[Ubiquitous Traffic Eyes](http://www.seutraffic.com/#/download)

[leveLXData](https://levelxdata.com/)

## Example statistical results:
The following table includes the performance between the standard PPO and baseline IDM/MOBIL model:

| metric | better | stock-ppo | IDM/MOBIL |
|---|---|---|---|
| Return | higher | 9.417 | 6.667 |
| Collision Rate | lower | 0.000 | 1.000 |
| TTC Min | higher | 0.622 | 0.000 |
| Criticality Rate | lower | 0.273 | 0.778 |
| Min Spacing | higher | 3.517 | 0.000 |
| Mean Speed | higher | 1.459 | 2.218 |
| Mean |Jerk| | lower | 0.513 | 2.529 |
| Final Progress | higher | 16.893 | 18.291 |
| Imposed Risk Potential | lower | 52.420 | 50.544 |
| Backward Risk Flux | lower | 0.000 | 0.000 |
| Interaction Density | lower | 0.418 | 0.511 |
| Mean Risk per Vehicle | lower | 0.020 | 0.077 |
| Backward Flux Ratio | lower | 0.180 | 0.279 |
| Frame Mean Speed | higher | 11.936 | 12.540 |
| Frame Speed Variance | lower | 24.828 | 13.549 |
| Frame Total Progress Rate | higher | 59.680 | 62.700 |
| Efficiency Index (EI) | higher | 0.738 | 0.847 |
| Safety-Efficiency Index (SEI) | higher | 0.738 | 0.764 |
| Social Traffic Efficiency Index | higher | 9.409 | 11.095 |
| Shockwave Onset Rate | lower | 0.182 | 0.111 |
| Frame Min TTC | higher | nan | 0.601 |
| Frame Frac TTC < 1.5s | lower | 0.000 | 0.111 |
| Frame Max DRAC | lower | 0.000 | 0.924 |
| Safety Score | higher | 0.810 | 0.301 |
| Courtesy Score | higher | 1.000 | 1.000 |
| Social-Friendliness Score | higher | 0.608 | 0.421 |

## Example snapshots of agent performances
Comparing the Social-friendly and risk-aware RL agent with the baseline RL and IDM/MOBIL in roundabout scenario: The IDM/MOBIL leads to a collision, while the baseline RL agent leads to over-conservative behavior
![Result](assests/roundabout_snapshot.jpg)

# MetaDrive RL Training, Evaluation, and Visualization Commands

This file is the command reference for the MetaDrive family of experiments in
`rl/`. It separates protocol-consistent paper runs from diagnostic visualization
overrides.

## Why Traffic Can Look Static

MetaDrive's default benchmark traffic mode is `trigger`. In trigger mode, traffic
vehicles are staged and activated only when the ego vehicle reaches the trigger
road. This can look static in short or conservative rollouts, especially on
`map="S"` straight-road specialists and some intersection seeds. It is not a
checkpoint-loading bug.

Use one of these fixes:

- For paper experiments with moving traffic, train and evaluate the explicit
  `*_respawn` protocols, for example `matched_social_risk_intersection_respawn`.
- For visual diagnosis of an old trigger-trained checkpoint, pass
  `--traffic-mode respawn` to the watch/eval/visualization scripts. Label this as
  a viewer/stress-test override unless the checkpoint was trained with respawn
  traffic.

Do not replace old checkpoint folders. Use new run names.

## Scenario Protocols

| Scenario | Map code | Trigger stock/risk | Moving stock/risk |
|---|---:|---|---|
| Straight | `S` | `matched_stock_straight`, `matched_social_risk_straight` | `matched_stock_straight_respawn`, `matched_social_risk_straight_respawn` |
| Curve | `C` | `matched_stock_curve`, `matched_social_risk_curve` | `matched_stock_curve_respawn`, `matched_social_risk_curve_respawn` |
| Merge | `r` | `matched_stock_merge`, `matched_social_risk_merge` | `matched_stock_merge_respawn`, `matched_social_risk_merge_respawn` |
| Intersection | `X` | `matched_stock_intersection`, `matched_social_risk_intersection` | `matched_stock_intersection_respawn`, `matched_social_risk_intersection_respawn` |
| Roundabout | `O` | `matched_stock_roundabout`, `matched_social_risk_roundabout` | `matched_stock_roundabout_respawn`, `matched_social_risk_roundabout_respawn` |
| Mixed | `SCrXO` | `matched_stock_mixed`, `matched_social_risk_mixed` | `matched_stock_mixed_respawn`, `matched_social_risk_mixed_respawn` |

Continuous-action counterparts append `_continuous` to trigger protocols and
`_respawn_continuous` to moving-traffic protocols. Use these for SAC, TD3, and
DDPG.

## Algorithm Compatibility

| Algorithm | SB3 action space | Use these protocols |
|---|---|---|
| PPO | Discrete or continuous | Main paper uses discrete `matched_*`; continuous is also supported |
| DQN | Single discrete action only | `matched_*` or `matched_*_respawn` |
| SAC | Continuous `Box([-1,1]^2)` | `matched_*_continuous` or `matched_*_respawn_continuous` |
| TD3 | Continuous `Box([-1,1]^2)` | `matched_*_continuous` or `matched_*_respawn_continuous` |
| DDPG | Continuous `Box([-1,1]^2)` | `matched_*_continuous` or `matched_*_respawn_continuous` |

Stock protocols use MetaDrive's default observation and reward. Social-risk
protocols append the 8-D DRIFT risk feature vector, apply risk/comfort reward
shaping, and compute risk exposure metrics.

## MetaDrive Training Commands
### PPO Specialist Runs With Moving Traffic

Straight:

```powershell
python rl/train_metadrive_sb3.py --protocol matched_stock_straight_respawn --algo ppo --steps 1000000 --n-envs 4 --run-name matched_stock_straight_respawn_ppo_1m
python rl/train_metadrive_sb3.py --protocol matched_social_risk_straight_respawn --algo ppo --steps 1000000 --n-envs 4 --run-name matched_social_risk_straight_respawn_ppo_1m
```

Intersection:

```powershell
python rl/train_metadrive_sb3.py --protocol matched_stock_intersection_respawn --algo ppo --steps 1000000 --n-envs 4 --run-name matched_stock_intersection_respawn_ppo_1m
python rl/train_metadrive_sb3.py --protocol matched_social_risk_intersection_respawn --algo ppo --steps 1000000 --n-envs 4 --run-name matched_social_risk_intersection_respawn_ppo_1m
```

Merge, roundabout, and curve follow the same naming pattern:

```powershell
python rl/train_metadrive_sb3.py --protocol matched_stock_merge_respawn --algo ppo --steps 1000000 --n-envs 4 --run-name matched_stock_merge_respawn_ppo_1m
python rl/train_metadrive_sb3.py --protocol matched_social_risk_merge_respawn --algo ppo --steps 1000000 --n-envs 4 --run-name matched_social_risk_merge_respawn_ppo_1m

python rl/train_metadrive_sb3.py --protocol matched_stock_roundabout_respawn --algo ppo --steps 1000000 --n-envs 4 --run-name matched_stock_roundabout_respawn_ppo_1m
python rl/train_metadrive_sb3.py --protocol matched_social_risk_roundabout_respawn --algo ppo --steps 1000000 --n-envs 4 --run-name matched_social_risk_roundabout_respawn_ppo_1m

python rl/train_metadrive_sb3.py --protocol matched_stock_curve_respawn --algo ppo --steps 1000000 --n-envs 4 --run-name matched_stock_curve_respawn_ppo_1m
python rl/train_metadrive_sb3.py --protocol matched_social_risk_curve_respawn --algo ppo --steps 1000000 --n-envs 4 --run-name matched_social_risk_curve_respawn_ppo_1m
```

Mixed-map generalization:

```powershell
python rl/train_metadrive_sb3.py --protocol matched_stock_mixed_respawn --algo ppo --steps 2000000 --n-envs 4 --run-name matched_stock_mixed_respawn_ppo_2m
python rl/train_metadrive_sb3.py --protocol matched_social_risk_mixed_respawn --algo ppo --steps 2000000 --n-envs 4 --run-name matched_social_risk_mixed_respawn_ppo_2m
```

CUDA is optional and should be added only when the local PyTorch installation is
CUDA-enabled:

```powershell
python rl/train_metadrive_sb3.py --protocol matched_social_risk_intersection_respawn --algo ppo --steps 1000000 --n-envs 4 --run-name matched_social_risk_intersection_respawn_ppo_1m_cuda --device cuda
```

### DQN Baselines

DQN uses the same discrete protocols as PPO:

```powershell
python rl/train_metadrive_sb3.py --protocol matched_stock_intersection_respawn --algo dqn --steps 1000000 --n-envs 4 --run-name matched_stock_intersection_respawn_dqn_1m
python rl/train_metadrive_sb3.py --protocol matched_social_risk_intersection_respawn --algo dqn --steps 1000000 --n-envs 4 --run-name matched_social_risk_intersection_respawn_dqn_1m
```

### SAC, TD3, and DDPG Baselines

Use continuous-action protocols:

```powershell
python rl/train_metadrive_sb3.py --protocol matched_stock_intersection_respawn_continuous --algo sac --steps 1000000 --n-envs 4 --run-name matched_stock_intersection_respawn_sac_1m
python rl/train_metadrive_sb3.py --protocol matched_social_risk_intersection_respawn_continuous --algo sac --steps 1000000 --n-envs 4 --run-name matched_social_risk_intersection_respawn_sac_1m

python rl/train_metadrive_sb3.py --protocol matched_stock_intersection_respawn_continuous --algo td3 --steps 1000000 --n-envs 4 --run-name matched_stock_intersection_respawn_td3_1m
python rl/train_metadrive_sb3.py --protocol matched_social_risk_intersection_respawn_continuous --algo td3 --steps 1000000 --n-envs 4 --run-name matched_social_risk_intersection_respawn_td3_1m

python rl/train_metadrive_sb3.py --protocol matched_stock_intersection_respawn_continuous --algo ddpg --steps 1000000 --n-envs 4 --run-name matched_stock_intersection_respawn_ddpg_1m
python rl/train_metadrive_sb3.py --protocol matched_social_risk_intersection_respawn_continuous --algo ddpg --steps 1000000 --n-envs 4 --run-name matched_social_risk_intersection_respawn_ddpg_1m
```

## Evaluation Commands

Planner specs use:

```text
label@protocol:path/to/final.zip
idm@protocol
random@protocol
```

The label suffix selects the SB3 loader: `_dqn`, `_sac`, `_td3`, `_ddpg`; no
suffix defaults to PPO.

PPO stock/risk/IDM comparison:

```powershell
python rl/eval_metadrive.py --run-name eval_intersection_respawn_ppo `
  --seeds 10000:10020 --densities 0.3 `
  --planners "stock_ppo@matched_stock_intersection_respawn:rl/checkpoints/metadrive/matched_stock_intersection_respawn_ppo_1m/final.zip,risk_ppo@matched_social_risk_intersection_respawn:rl/checkpoints/metadrive/matched_social_risk_intersection_respawn_ppo_1m/final.zip,idm@matched_stock_intersection_respawn"
```

Multi-algorithm comparison:

```powershell
python rl/eval_metadrive.py --run-name eval_intersection_respawn_all_algos `
  --seeds 10000:10020 --densities 0.3 `
  --planners "stock_ppo@matched_stock_intersection_respawn:rl/checkpoints/metadrive/matched_stock_intersection_respawn_ppo_1m/final.zip,risk_ppo@matched_social_risk_intersection_respawn:rl/checkpoints/metadrive/matched_social_risk_intersection_respawn_ppo_1m/final.zip,stock_dqn@matched_stock_intersection_respawn:rl/checkpoints/metadrive/matched_stock_intersection_respawn_dqn_1m/final.zip,risk_dqn@matched_social_risk_intersection_respawn:rl/checkpoints/metadrive/matched_social_risk_intersection_respawn_dqn_1m/final.zip,stock_sac@matched_stock_intersection_respawn_continuous:rl/checkpoints/metadrive/matched_stock_intersection_respawn_sac_1m/final.zip,risk_sac@matched_social_risk_intersection_respawn_continuous:rl/checkpoints/metadrive/matched_social_risk_intersection_respawn_sac_1m/final.zip,idm@matched_stock_intersection_respawn"
```

If evaluating an old trigger-trained checkpoint under moving traffic only for
diagnosis:

```powershell
python rl/eval_metadrive.py --run-name eval_trigger_checkpoint_respawn_stress `
  --traffic-mode respawn --seeds 10000:10005 --densities 0.3 `
  --planners "risk_ppo@matched_social_risk_intersection:rl/checkpoints/metadrive/matched_social_risk_intersection_ppo_1m/final.zip,idm@matched_stock_intersection"
```

## 3D and Top-Down Watching

3D Panda3D view:

```powershell
python rl/watch_metadrive_agent.py --planner rl --algo ppo `
  --protocol matched_social_risk_intersection_respawn `
  --checkpoint rl/checkpoints/metadrive/matched_social_risk_intersection_respawn_ppo_1m/final.zip `
  --view 3d --episodes 3 --seed 10000 --density 0.3
```

Top-down viewer:

```powershell
python rl/watch_metadrive_agent.py --planner rl --algo ppo `
  --protocol matched_social_risk_intersection_respawn `
  --checkpoint rl/checkpoints/metadrive/matched_social_risk_intersection_respawn_ppo_1m/final.zip `
  --view top_down --episodes 3 --seed 10000 --density 0.3
```

Diagnostic workaround for static trigger traffic in an old checkpoint:

```powershell
python rl/watch_metadrive_agent.py --planner rl --algo ppo `
  --protocol matched_social_risk_intersection `
  --checkpoint rl/checkpoints/metadrive/matched_social_risk_intersection_ppo_1m/final.zip `
  --traffic-mode respawn --view 3d --episodes 3 --seed 10000 --density 0.3
```

IDM in the same moving-traffic scenario:

```powershell
python rl/watch_metadrive_agent.py --planner idm `
  --protocol matched_stock_intersection_respawn `
  --view 3d --episodes 3 --seed 10000 --density 0.3
```

## BEV Risk-Field Overlay

Side-by-side top-down risk overlay:

```powershell
python rl/visualize_metadrive_comparison.py `
  --planners "stock_ppo@matched_stock_intersection_respawn:rl/checkpoints/metadrive/matched_stock_intersection_respawn_ppo_1m/final.zip,risk_ppo@matched_social_risk_intersection_respawn:rl/checkpoints/metadrive/matched_social_risk_intersection_respawn_ppo_1m/final.zip,idm@matched_stock_intersection_respawn" `
  --seed 10000 --density 0.3 --max-steps 200 --step-stride 20 `
  --out rl/logs/metadrive/viz/intersection_respawn_overlay.png
```

