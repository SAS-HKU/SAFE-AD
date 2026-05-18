
# Risk-Informed Physics-Based Policy Learning for Externality-Aware Autonomous Driving

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

## Set1: HighwayEnv Configurations
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

### Datasets used in this project (download links):
[Ubiquitous Traffic Eyes](http://www.seutraffic.com/#/download)

[leveLXData](https://levelxdata.com/)


### Example snapshots of agent performances
Comparing the Social-friendly and risk-aware RL agent with the baseline RL and IDM/MOBIL in roundabout scenario: The IDM/MOBIL leads to a collision, while the baseline RL agent leads to over-conservative behavior
![Result](assests/roundabout_snapshot.jpg)


## Set2: MetaDrive Configurations

### Algorithm Compatibility

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

### Reward Profiles

The old successful social-risk checkpoint used the stable risk-only shaping:

```text
r_t = r_t^MD - lambda_R R_t(0,0)
```

Use this as the main training profile first:

```powershell
--reward-profile risk_only
```

Comfort-aware training is an ablation after the moving policy is working:

```powershell
--reward-profile comfort_light
--reward-profile risk_comfort
```

If a social-risk specialist policy keeps the ego static, check whether it was
trained with the full comfort profile. The static policy usually outputs a
low/no-throttle discrete action for the whole episode and obtains low negative
reward with almost zero route completion.

### Training Commands

### Official MetaDrive Notebook Reference

This reproduces the tutorial-style PPO sanity check. It is not the main fair
benchmark.

```powershell
python rl/train_metadrive_stock.py
```

#### PPO Specialist Runs With Moving Traffic

Straight:

```powershell
python rl/train_metadrive_sb3.py --protocol matched_stock_straight_respawn --algo ppo --steps 1000000 --n-envs 4 --run-name matched_stock_straight_respawn_ppo_1m
python rl/train_metadrive_sb3.py --protocol matched_social_risk_straight_respawn --algo ppo --steps 1000000 --n-envs 4 --reward-profile risk_only --run-name matched_social_risk_straight_respawn_ppo_1m
```

Intersection:

```powershell
python rl/train_metadrive_sb3.py --protocol matched_stock_intersection_respawn --algo ppo --steps 1000000 --n-envs 4 --run-name matched_stock_intersection_respawn_ppo_1m
python rl/train_metadrive_sb3.py --protocol matched_social_risk_intersection_respawn --algo ppo --steps 1000000 --n-envs 4 --reward-profile risk_only --run-name matched_social_risk_intersection_respawn_ppo_1m
```

Merge, roundabout, and curve follow the same naming pattern:

```powershell
python rl/train_metadrive_sb3.py --protocol matched_stock_merge_respawn --algo ppo --steps 1000000 --n-envs 4 --run-name matched_stock_merge_respawn_ppo_1m
python rl/train_metadrive_sb3.py --protocol matched_social_risk_merge_respawn --algo ppo --steps 1000000 --n-envs 4 --reward-profile risk_only --run-name matched_social_risk_merge_respawn_ppo_1m

python rl/train_metadrive_sb3.py --protocol matched_stock_roundabout_respawn --algo ppo --steps 1000000 --n-envs 4 --run-name matched_stock_roundabout_respawn_ppo_1m
python rl/train_metadrive_sb3.py --protocol matched_social_risk_roundabout_respawn --algo ppo --steps 1000000 --n-envs 4 --reward-profile risk_only --run-name matched_social_risk_roundabout_respawn_ppo_1m

python rl/train_metadrive_sb3.py --protocol matched_stock_curve_respawn --algo ppo --steps 1000000 --n-envs 4 --run-name matched_stock_curve_respawn_ppo_1m
python rl/train_metadrive_sb3.py --protocol matched_social_risk_curve_respawn --algo ppo --steps 1000000 --n-envs 4 --reward-profile risk_only --run-name matched_social_risk_curve_respawn_ppo_1m
```

Mixed-map generalization:

```powershell
python rl/train_metadrive_sb3.py --protocol matched_stock_mixed_respawn --algo ppo --steps 2000000 --n-envs 4 --run-name matched_stock_mixed_respawn_ppo_2m
python rl/train_metadrive_sb3.py --protocol matched_social_risk_mixed_respawn --algo ppo --steps 2000000 --n-envs 4 --reward-profile risk_only --run-name matched_social_risk_mixed_respawn_ppo_2m
```

CUDA is optional:

```powershell
python rl/train_metadrive_sb3.py --protocol matched_social_risk_intersection_respawn --algo ppo --steps 1000000 --n-envs 4 --reward-profile risk_only --run-name matched_social_risk_intersection_respawn_ppo_1m_cuda --device cuda
```

![PPO_examples](assests/ppo_trial.gif)

#### DQN Baselines

DQN uses the same discrete protocols as PPO:

```powershell
python rl/train_metadrive_sb3.py --protocol matched_stock_intersection_respawn --algo dqn --steps 1000000 --n-envs 4 --run-name matched_stock_intersection_respawn_dqn_1m
python rl/train_metadrive_sb3.py --protocol matched_social_risk_intersection_respawn --algo dqn --steps 1000000 --n-envs 4 --reward-profile risk_only --run-name matched_social_risk_intersection_respawn_dqn_1m
```

#### SAC, TD3, and DDPG Baselines

Use continuous-action protocols:

```powershell
python rl/train_metadrive_sb3.py --protocol matched_stock_intersection_respawn_continuous --algo sac --steps 1000000 --n-envs 4 --run-name matched_stock_intersection_respawn_sac_1m
python rl/train_metadrive_sb3.py --protocol matched_social_risk_intersection_respawn_continuous --algo sac --steps 1000000 --n-envs 4 --run-name matched_social_risk_intersection_respawn_sac_1m

python rl/train_metadrive_sb3.py --protocol matched_stock_intersection_respawn_continuous --algo td3 --steps 1000000 --n-envs 4 --run-name matched_stock_intersection_respawn_td3_1m
python rl/train_metadrive_sb3.py --protocol matched_social_risk_intersection_respawn_continuous --algo td3 --steps 1000000 --n-envs 4 --run-name matched_social_risk_intersection_respawn_td3_1m

python rl/train_metadrive_sb3.py --protocol matched_stock_intersection_respawn_continuous --algo ddpg --steps 1000000 --n-envs 4 --run-name matched_stock_intersection_respawn_ddpg_1m
python rl/train_metadrive_sb3.py --protocol matched_social_risk_intersection_respawn_continuous --algo ddpg --steps 1000000 --n-envs 4 --run-name matched_social_risk_intersection_respawn_ddpg_1m
```

#### Evaluation Commands

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

#### 3D and Top-Down Watching

3D Panda3D view:

```powershell
python rl/watch_metadrive_agent.py --planner rl --algo ppo `
  --protocol matched_social_risk_intersection_respawn `
  --checkpoint rl/checkpoints/metadrive/matched_social_risk_intersection_respawn_ppo_1m/final.zip `
  --view 3d --episodes 3 --seed 10000 --density 0.3
```

Debug a suspected static policy:

```powershell
python rl/watch_metadrive_agent.py --planner rl --algo ppo `
  --protocol matched_social_risk_intersection `
  --checkpoint rl/checkpoints/metadrive/matched_social_risk_intersection_ppo_1m/final.zip `
  --view none --episodes 1 --seed 10000 --density 0.3 `
  --debug-actions --debug-obs-tail --no-realtime --max-steps 50
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

#### BEV Risk-Field Overlay

Side-by-side top-down risk overlay:

```powershell
python rl/visualize_metadrive_comparison.py `
  --planners "stock_ppo@matched_stock_intersection_respawn:rl/checkpoints/metadrive/matched_stock_intersection_respawn_ppo_1m/final.zip,risk_ppo@matched_social_risk_intersection_respawn:rl/checkpoints/metadrive/matched_social_risk_intersection_respawn_ppo_1m/final.zip,idm@matched_stock_intersection_respawn" `
  --seed 10000 --density 0.3 --max-steps 200 --step-stride 20 `
  --out rl/logs/metadrive/viz/intersection_respawn_overlay.png
```

