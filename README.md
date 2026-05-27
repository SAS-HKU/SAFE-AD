# SAFE-AD: Socially-Aware Field-Enhanced Reinforcement Learning for Autonomous Driving
Zian Wang, Wenjie Huang, Zejian Deng, Yong Wang, Jiahui Xu, Yiming Shu, Shen Li, Dongpu Cao, Chen Sun✉

![Code Status](https://img.shields.io/badge/code-partial_release-orange)
![Demos](https://img.shields.io/badge/demonstrations-ready-brightgreen)
![Preprint](https://img.shields.io/badge/preprint-coming_soon-lightgrey)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)

SAFE-AD is a research prototype for **socially-aware and risk-aware reinforcement learning in interactive autonomous driving**.  
The central idea is to use a **physics-informed propagated risk field** as a structured intermediate representation for RL tactical planning. Instead of penalizing only instantaneous scalar risk, SAFE-AD models how risk propagates through traffic and maps this field to ego safety, surrounding-vehicle exposure, and social externality.

The preliminary PDE-governed risk-field model is based on [DRIFT](https://github.com/PeterWANGHK/DRIFT.git).

![Methodology graph](assests/SAFE-AD-v1.png)

## Project Status

| Component | Status |
|---|---|
| Demonstration files | Ready |
| Highway-env experiments | Demo-ready |
| MetaDrive experiments | Demo-ready |
| Full code release | In preparation |
| Paper preprint | Coming soon |
| Checkpoints and full logs | Partial release / in preparation |

## Overview

SAFER evaluates field-enhanced RL in three complementary settings:

| Layer | Purpose | Environment |
|---|---|---|
| Risk-field modeling | Validate propagated risk and PINN surrogate modeling | Naturalistic trajectory datasets and synthetic traffic scenes |
| Controlled RL benchmark | Diagnose whether field features improve tactical behavior | highway-env |
| High-fidelity RL benchmark | Test robustness under richer vehicle dynamics and interactive traffic | MetaDrive |
| Safety-critical execution | Use learned RL guidance with MPC-CBF safety filtering | Synthetic merging / uncertainty scenarios |

The benchmark compares stock RL, risk-aware RL, socially-aware risk RL, IDM/MOBIL, and selected SB3 baselines.

## Core Ideas

- **Propagated risk field**: models spatial-temporal traffic risk instead of only instantaneous ego risk.
- **PINN surrogate**: learns a differentiable approximation of the PDE-governed risk field.
- **Risk-aware RL**: appends field-derived risk features to the policy observation.
- **Social-aware reward shaping**: penalizes imposed risk, backward disturbance, jerk, abrupt steering, and unsafe close interactions.
- **MPC-CBF compatibility**: learned RL guidance can be used as a tactical layer while MPC-CBF enforces hard safety constraints.

## Risk Field and PINN Demonstrations

Numerically solved risk field and PINN-generated risk field:

![PINN_examples](assests/DRIFT_PINN_1.gif)

PINN field outputs across highway, merging, roundabout, and intersection scenarios:

![PINN_scenario](assests/pinn_result.jpg)

## Datasets

The project uses naturalistic driving datasets for trajectory processing, behavior extraction, and field validation.

Dataset sources:

- [Ubiquitous Traffic Eyes](http://www.seutraffic.com/#/download)
- [leveLXData](https://levelxdata.com/)

Example dataset extraction:

```bash
python -m rl.data.historical_extractor \
  --dataset SQM-N-4 \
  --data-dir data/SQM-N-4 \
  --out-path rl/checkpoints/bc_sqm_v3.npz
```

## Highway-env Experiments

The highway-env layer is used as a controlled and interpretable benchmark.  
It evaluates whether risk-field and social-interaction features improve tactical RL behavior in highway, merge, roundabout, and intersection scenarios.

The environment configurations are forked from [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv.git).

Example training workflow:

```bash
python -m rl.data.historical_extractor \
  --data-dir data/exiD \
  --recordings all \
  --out-path rl/checkpoints/bc_dataset_full.npz \
  --horizon-sec 1.5

python -m rl.train_bc \
  --dataset rl/checkpoints/bc_dataset_full.npz \
  --out rl/checkpoints/decision_policy_bc.pt

python -m rl.train_decision_ppo \
  --bc-checkpoint rl/checkpoints/decision_policy_bc.pt \
  --out rl/checkpoints/decision_policy_ppo.pt \
  --total-steps 200000
```

Example evaluation:

```bash
python highway_test.py \
  --rl-policy-mode decision \
  --rl-decision-checkpoint rl/checkpoints/decision_policy_ppo.pt \
  --steps 400 \
  --save-dir figsave_test_rl
```

Example snapshot comparing a social/risk-aware RL agent with baseline RL and IDM/MOBIL:

![Result](assests/roundabout_snapshot.jpg)

## MetaDrive Experiments

The MetaDrive layer tests whether the same field-enhanced RL design transfers to higher-fidelity driving with procedural maps, continuous vehicle dynamics, and interactive IDM traffic.

The environment configurations are forked from [MetaDrive](https://github.com/metadriverse/metadrive.git).

### Algorithm Support

| Algorithm | Action Space | Usage |
|---|---|---|
| PPO | Discrete / continuous | Main discrete benchmark |
| DQN | Discrete | Discrete baseline |
| SAC | Continuous | Continuous-control baseline |
| TD3 | Continuous | Continuous-control baseline |
| DDPG | Continuous | Continuous-control baseline |
| IDM/MOBIL | Rule-based | Reference controller |

### Training Template

```bash
python rl/train_metadrive_sb3.py \
  --protocol matched_stock_intersection_respawn \
  --algo ppo \
  --steps 1000000 \
  --n-envs 4 \
  --run-name matched_stock_intersection_respawn_ppo_1m
```

Risk-aware or social-risk variants use the matched social-risk protocol:

```bash
python rl/train_metadrive_sb3.py \
  --protocol matched_social_risk_intersection_respawn \
  --algo ppo \
  --steps 1000000 \
  --n-envs 4 \
  --reward-profile risk_only \
  --run-name matched_social_risk_intersection_respawn_ppo_1m
```

CUDA is optional:

```bash
python rl/train_metadrive_sb3.py \
  --protocol matched_social_risk_intersection_respawn \
  --algo ppo \
  --steps 1000000 \
  --n-envs 4 \
  --device cuda
```

Continuous-control algorithms use continuous protocols:

```bash
python rl/train_metadrive_sb3.py \
  --protocol matched_stock_intersection_respawn_continuous \
  --algo sac \
  --steps 1000000 \
  --n-envs 4 \
  --run-name matched_stock_intersection_respawn_sac_1m
```

## Evaluation

MetaDrive evaluation supports stock RL, risk-aware RL, social-risk RL, IDM, and random baselines.

Planner format:

```text
label@protocol:path/to/final.zip
idm@protocol
random@protocol
```

Example:

```bash
python rl/eval_metadrive.py \
  --run-name eval_intersection_respawn_ppo \
  --seeds 10000:10020 \
  --densities 0.3 \
  --planners "stock_ppo@matched_stock_intersection_respawn:rl/checkpoints/metadrive/matched_stock_intersection_respawn_ppo_1m/final.zip,risk_ppo@matched_social_risk_intersection_respawn:rl/checkpoints/metadrive/matched_social_risk_intersection_respawn_ppo_1m/final.zip,idm@matched_stock_intersection_respawn"
```

Main reported metrics are grouped into four pillars:

| Pillar | Metrics |
|---|---|
| Task performance | success rate, route completion, episode return |
| Safety and risk | collision rate, out-of-road rate, TTC violation, near-miss count, cumulative risk exposure, peak risk |
| Efficiency and flow | mean speed, progress, EI, SEI, SEMI |
| Comfort and sociality | jerk, steering-change rate, throttle-change rate, backward disturbance, imposed risk, social score |

## Visualization

3D simulator view:

```bash
python rl/watch_metadrive_agent.py \
  --planner rl \
  --algo ppo \
  --protocol matched_social_risk_intersection_respawn \
  --checkpoint rl/checkpoints/metadrive/matched_social_risk_intersection_respawn_ppo_1m/final.zip \
  --view 3d \
  --episodes 3 \
  --seed 10000 \
  --density 0.3
```

Top-down view with risk-field overlay:

```bash
python rl/watch_metadrive_agent.py \
  --planner rl \
  --protocol matched_stock_merge_respawn \
  --checkpoint rl/checkpoints/metadrive/matched_stock_merge_respawn_ppo_1m/final.zip \
  --view top_down \
  --risk-overlay drift \
  --episodes 3 \
  --seed 10000 \
  --density 0.3
```

Example baseline training behavior in MetaDrive:

![TD3_examples](assests/stock_td3_intersection.gif)

![DDPG_examples](assests/stock_ddpg_intersection.gif)

![PPO_examples](assests/stock_roundabout_ppo.gif)

Example proposed and baseline policy rollouts:

![PPO_examples](assests/social_risk_ppo_intersection.gif)

![DDPG_examples](assests/ddpg_roundabout.gif)

![SAC_parallel_examples](assests/intersection_sac_parallel.gif)


