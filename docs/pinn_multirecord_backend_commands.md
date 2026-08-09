# Prospective PINN Validation and Policy-Backend Swap

This workflow keeps three questions separate:

1. Does the numerical field rank future interaction occupancy better than its
   instantaneous source?
2. Does a recording-disjoint, context-conditioned PINN preserve that field?
3. Does replacing the numerical backend with the PINN preserve the actions and
   closed-loop outcomes of one fixed field-observing policy?

The PINN is accepted as a deployment backend only if all three checks pass.
Existing paper checkpoints are not overwritten.

## 1. Cache the causal prospective teacher

The teacher uses only the state available at the current sensing instant. It
integrates source transport and diffusion over a 3 s horizon with 0.25 s
quadrature steps. Future naturalistic trajectories are used later as labels,
never as teacher inputs.

```powershell
python -m rl.build_prospective_pinn_cache `
  --input-cache-globs `
    "evaluation/pinn_teacher_cache/inD_*" `
    "evaluation/pinn_highway_teacher_cache/highwayenv_*" `
  --output-root evaluation/pinn_prospective_v2_cache `
  --horizon-s 3 --integration-step-s 0.25 `
  --decay-rate 0.25 --transport-scale 1 `
  --x-min -40 --x-max 100 --nx 141 `
  --y-min -20 --y-max 20 --ny 41
```

Completed caches are restored from their manifests. Use `--rebuild` only when
the declared solver or grid has intentionally changed.

## 2. Train the multi-recording, domain-conditioned PINN

Calibration data comprise inD recordings 01-05 and HighwayEnv seeds 0-2 from
highway, merge, intersection, and roundabout. Held-out evaluation uses inD
06-11 and HighwayEnv seeds 100-102. Domain normalization is inferred only from
the calibration split.

```powershell
python -m rl.train_prospective_context_pinn --stage all `
  --calibration-cache-globs `
    "evaluation/pinn_prospective_v2_cache/inD_0[1-5]" `
    "evaluation/pinn_prospective_v2_cache/highwayenv_*_seed000[0-2]" `
  --heldout-cache-globs `
    "evaluation/pinn_prospective_v2_cache/inD_0[6-9]" `
    "evaluation/pinn_prospective_v2_cache/inD_1[0-1]" `
    "evaluation/pinn_prospective_v2_cache/highwayenv_*_seed010[0-2]" `
  --output-dir evaluation/pinn_prospective_context_v3_domain_conditioned `
  --model-out rl/checkpoints/pinn/pinn_prospective_context_v3_domain_conditioned.pt `
  --device cpu --torch-threads 2 --seed 2026 `
  --steps 6000 --batch-size 8 --crop-width 128 `
  --risk-patch-fraction 0.65 --width 16 `
  --dilations 1,2,4,8,16,1 --lr 0.0008 `
  --hotspot-boost 4 --w-field 1 --w-gradient 0.30 `
  --w-physics 0.002 --validation-frames 50
```

## 3. Validate future occupancy and numerical fidelity

```powershell
python -m rl.evaluate_prospective_field_validity `
  --dataset inD --data-root data `
  --cache-root evaluation/pinn_prospective_v2_cache `
  --recordings 06,07,08,09,10,11 `
  --horizon-s 3 --sample-step-s 0.2 --frame-stride 10 `
  --pinn-checkpoint rl/checkpoints/pinn/pinn_prospective_context_v3_domain_conditioned.pt `
  --device cpu --bootstrap 5000 `
  --output-dir evaluation/prospective_field_validity_pinn_v3

python -m rl.plot_prospective_pinn_validation `
  --validity-dir evaluation/prospective_field_validity_pinn_v3 `
  --fidelity-dir evaluation/pinn_prospective_context_v3_domain_conditioned `
  --output evaluation/pinn_prospective_v3_paper/pinn_validity_runtime
```

Report recording-bootstrap confidence intervals for future-occupancy AUROC and
AUPRC. Field RMSE, correlation, gradient angle, hotspot overlap, and timing are
diagnostic quantities; cached frames are not treated as independent samples.

## 4. Train a policy that actually observes the field

The stock flattened observation has 25 entries. `--append-risk-obs true`
appends eight field descriptors, yielding the 33-D policy required for a valid
backend swap.

```powershell
python -m rl.train_highwayenv_social_sb3 `
  --algo ppo --env-id merge-v0 --eval-env-id merge-v0 `
  --interface stock --action-mode continuous `
  --reward-config rl/config/social_reward_v1.json --ablation A5 `
  --use-drift true --append-risk-obs true `
  --field-backend prospective `
  --pinn-checkpoint rl/checkpoints/pinn/pinn_prospective_context_v3_domain_conditioned.pt `
  --traffic-preset medium --total-steps 20000 `
  --eval-freq 4000 --eval-episodes 3 --eval-max-steps 600 `
  --n-envs 2 --seed 2026 `
  --run-dir rl/logs/revision_merge_ppo_continuous_fieldobs33_prospective_v3_20k_bounded `
  --device cpu
```

`merge-v0` does not impose a native time truncation in every installed
HighwayEnv version. The explicit 600-step bound is therefore part of the
evaluation protocol, not an early-stopping optimization.

## 5. Run the actual teacher-to-PINN swap

```powershell
python -m rl.eval_highwayenv_backend_swap_sb3 `
  --algo ppo `
  --policy-checkpoint `
    rl/logs/revision_merge_ppo_continuous_fieldobs33_prospective_v3_20k_bounded/checkpoints/best_model.zip `
  --pinn-checkpoint `
    rl/checkpoints/pinn/pinn_prospective_context_v3_domain_conditioned.pt `
  --env-id merge-v0 --action-mode continuous --ablation A5 `
  --traffic-preset medium --seeds 100:110 --pinn-device cpu `
  --max-episode-steps 600 --bootstrap 5000 `
  --output-dir evaluation/highwayenv_prospective_pinn_backend_swap_v3
```

The evaluator first keeps both simulators on the teacher trajectory and
compares field descriptors and counterfactual actions at identical states. It
then runs paired closed-loop episodes with either backend. Report policy
inference, field backend, environment stepping, and rendering separately.

### Runtime decomposition

Run the numerical solver and PINN on the same held-out ego-local coefficient
snapshots. The generated table separates context construction, neural/PDE
kernel, device transfer, and complete field-core latency at batch size one.

```powershell
python -m rl.benchmark_prospective_field_runtime `
  --cache-globs `
    "evaluation/pinn_prospective_v2_cache/highwayenv_merge_v0_seed010*" `
  --checkpoint `
    rl/checkpoints/pinn/pinn_prospective_context_v3_domain_conditioned.pt `
  --devices cpu cuda --frames-per-recording 20 `
  --torch-threads 2 `
  --output-dir evaluation/pinn_prospective_runtime_v3
```

The field-core benchmark starts after ego-local scene conditioning. The
closed-loop backend-swap table remains the authoritative online latency test.
Do not combine either value with rendering time or one-time model loading.

### Four-scenario planning comparison

The merge checkpoint above cannot be reused as a four-scenario headline result.
Highway, merge, and roundabout expose a 33-D field observation, whereas the
intersection exposes 113 entries (105 base + the same eight field descriptors).
Train one matched continuous-control specialist per scenario and keep the
prospective numerical backend fixed during training. This lets evaluation
replace only the field-query backend while holding the policy, traffic seed,
reward, and action space fixed.

The following SAC runs should start only after the active MetaDrive CUDA batch
has completed:

```powershell
$scenarios = @{
  highway     = "highway-v0"
  merge       = "merge-v0"
  intersection = "intersection-v0"
  roundabout  = "roundabout-v0"
}

foreach ($name in $scenarios.Keys) {
  $envId = $scenarios[$name]
  python -m rl.train_highwayenv_social_sb3 `
    --algo sac --env-id $envId --eval-env-id $envId `
    --interface stock --action-mode continuous `
    --reward-config rl/config/social_reward_v1.json --ablation A5 `
    --use-drift true --append-risk-obs true `
    --field-backend prospective --traffic-preset medium `
    --total-steps 500000 --eval-freq 25000 --eval-episodes 5 `
    --eval-max-steps 600 --n-envs 1 --seed 2026 `
    --run-dir "rl/logs/revision_${name}_sac_fieldobs_prospective_500k" `
    --device cuda
}
```

Evaluate each specialist on paired, unseen traffic seeds:

```powershell
$scenarios = @{
  highway     = "highway-v0"
  merge       = "merge-v0"
  intersection = "intersection-v0"
  roundabout  = "roundabout-v0"
}

foreach ($name in $scenarios.Keys) {
  $envId = $scenarios[$name]
  python -m rl.eval_highwayenv_backend_swap_sb3 `
    --algo sac `
    --policy-checkpoint "rl/logs/revision_${name}_sac_fieldobs_prospective_500k/checkpoints/best_model.zip" `
    --pinn-checkpoint rl/checkpoints/pinn/pinn_prospective_context_v3_domain_conditioned.pt `
    --env-id $envId --action-mode continuous --ablation A5 `
    --traffic-preset medium --seeds 100:130 --pinn-device cpu `
    --max-episode-steps 600 --bootstrap 5000 `
    --output-dir "evaluation/highwayenv_pinn_planning_${name}"
}
```

Aggregate the four paired episode files only after all scenarios are present:

```powershell
python -m rl.plot_highwayenv_pinn_planning_results `
  --episodes-csv "evaluation/highwayenv_pinn_planning_*/backend_swap_episodes.csv" `
  --bootstrap 5000 --permutations 20000 `
  --output-dir evaluation/highwayenv_pinn_planning_comparison
```

The aggregate table reports paired confidence intervals, randomization tests,
and Holm-adjusted p-values. An efficiency claim requires higher paired route
progress while collision, TTC, jerk, and imposed follower deceleration remain
non-degraded. A visually smaller PINN field is not itself evidence of better
planning.

For a stage-resolved online check without policy or rendering cost, run:

```powershell
python -m rl.benchmark_highwayenv_field_backend_online `
  --env-id merge-v0 --traffic-preset medium --seeds 100:103 `
  --devices cpu cuda --steps 40 --discard-first 5 `
  --checkpoint `
    rl/checkpoints/pinn/pinn_prospective_context_v3_domain_conditioned.pt `
  --output-dir evaluation/pinn_prospective_runtime_online_v3
```

## 6. Generate blue-to-red BEV animations

```powershell
python -m rl.visualize_dataset_pinn_gif `
  --dataset inD --recording 06 --data-root data `
  --cache-root evaluation/pinn_prospective_v2_cache `
  --checkpoint rl/checkpoints/pinn/pinn_prospective_context_v3_domain_conditioned.pt `
  --frames 60 --stride 4 --fps 8 `
  --output evaluation/pinn_prospective_v3_gifs/inD06_teacher_vs_pinn.gif

python -m rl.visualize_highwayenv_pinn_swap_gif `
  --env-id merge-v0 --planner idm `
  --action-mode default `
  --pinn-checkpoint `
    rl/checkpoints/pinn/pinn_prospective_context_v3_domain_conditioned.pt `
  --teacher-backend prospective --append-risk-obs false `
  --traffic-preset medium --seed 100 --frames 40 --fps 6 `
  --snapshot-index 8 --stop-on-offroad true `
  --output evaluation/pinn_prospective_v3_gifs/merge_idm_teacher_vs_pinn.gif
```

The qualitative merge figure deliberately uses HighwayEnv's IDM ego on the
standard merge road. It visualizes the numerical-to-PINN field replacement on
a lane-valid car-following/merging trajectory and is not presented as an RL
performance result. The separate backend-swap evaluation above remains the
policy-level test. This separation prevents an unstable policy trajectory
from being misinterpreted as a field-model failure.

Replace `merge-v0` with `roundabout-v0`, `intersection-v0`, or `highway-v0`
for scenario-specific demonstrations. All overlays use the `turbo` blue-to-red
scale: blue is low risk and red is high risk.

## 7. Build the revision figures

Build the naturalistic replay through the official track-import coordinate
pipeline. The script rotates each ego-local cached field back to the recorded
world pose and then applies the orthophoto calibration used for vehicle
bounding boxes:

```powershell
python -m rl.plot_dataset_pinn_orthophoto `
  --checkpoint rl/checkpoints/pinn/pinn_prospective_context_v3_domain_conditioned.pt `
  --device cpu `
  --data-root data `
  --output-dir evaluation/pinn_revision_figures_v5
```

It produces `pinn_naturalistic_orthophoto_overlay.pdf` in SciencePlots style,
together with a manifest containing the checkpoint hash, official recording
and frame identifiers, ego track, display calibration, and selection rule. The
figure contains only numerical/PINN scene overlays; it deliberately omits
absolute-error maps and fidelity statistics. Quantitative teacher-to-PINN
claims belong in the independent construct-validity table, while the practical
comparison belongs in the paired planning evaluation above.

To create the optional rounD/exiD external-domain panels, first cache and
ego-align the desired recordings, then build the prospective teacher:

```powershell
python -m rl.train_context_pinn --stage cache --dataset rounD `
  --data-root data --calibration-recordings 00 --heldout-recordings 01 `
  --cache-dir evaluation/pinn_teacher_cache_multidataset_v4 `
  --max-sec 10 --warmup-sec 2 `
  --output-dir evaluation/_unused_round_cache

python -m rl.train_context_pinn --stage cache --dataset exiD `
  --data-root data --calibration-recordings 00 --heldout-recordings 01 `
  --cache-dir evaluation/pinn_teacher_cache_multidataset_v4 `
  --max-sec 10 --warmup-sec 2 `
  --output-dir evaluation/_unused_exid_cache

python -m rl.augment_temporal_pinn_cache --cache-globs `
  "evaluation/pinn_teacher_cache_multidataset_v4/rounD_*" `
  "evaluation/pinn_teacher_cache_multidataset_v4/exiD_*" --rebuild

python -m rl.build_prospective_pinn_cache --input-cache-globs `
  "evaluation/pinn_teacher_cache_multidataset_v4/rounD_*" `
  "evaluation/pinn_teacher_cache_multidataset_v4/exiD_*" `
  --output-root evaluation/pinn_prospective_multidataset_v4 --rebuild
```

The accepted v3 checkpoint was calibrated on inD and HighwayEnv, not on rounD
or exiD. Therefore these two panels are zero-shot external diagnostics. Do not
pool them into the main fidelity estimate or call them cross-dataset validation
unless a new recording-disjoint multi-dataset checkpoint is trained and tested.

## Acceptance rule

Do not describe the PINN as a drop-in replacement if it fails independent
future-occupancy validity, materially rotates held-out gradients, changes the
frozen policy actions beyond the declared tolerance, or changes paired safety
and progress outcomes. In that case, retain the numerical backend for policy
results and report the PINN only as an approximation study.

## Verified results from the accepted run

The prospective teacher passes the independent construct-validity test on
held-out inD recordings 06-11. Relative to the instantaneous-source control,
it improves future-occupancy AUROC by 0.0848 (95% recording-bootstrap CI
0.0627-0.1005) and AUPRC by 0.1669 (0.1440-0.1919). The frozen PINN retains
gains of 0.0408 (0.0190-0.0576) and 0.0976 (0.0720-0.1207), respectively.

Across held-out HighwayEnv frames, PINN-to-teacher correlation is 0.925 and
the active-gradient angular error is 13.8 degrees. Merge is the strongest
topology-specific result: correlation 0.977, gradient error 7.4 degrees, and
equal-mass hotspot IoU 0.854. Roundabout remains the weakest case because many
held-out frames contain an almost zero teacher field; report its background
false-risk level alongside overlap rather than interpreting IoU alone.

The actual merge backend swap uses one frozen 33-D continuous PPO policy and
ten paired held-out traffic seeds. Synchronized replay gives descriptor RMSE
0.0047, mean action-vector error 6.68e-5, and no action error above 0.05. Both
closed-loop backends complete all ten episodes without collision; the
PINN-minus-teacher progress difference is -0.061 m (95% paired-bootstrap CI
-0.076 to -0.046 m) over approximately 586 m.

The timing result does **not** support an unconditional end-to-end acceleration
claim. In the paired field-core benchmark after ego-local conditioning, the
CUDA PINN takes 2.04 ms versus 2.64 ms for CPU numerical propagation. In the
complete online query, however, scene/context construction and device transfer
increase CUDA PINN latency to 5.80 ms versus 5.21 ms for the CPU numerical
backend. Until context construction is kept on-device, cached, batched, or
asynchronous, describe the PINN as a behaviorally interchangeable surrogate
with a faster GPU kernel, not as a uniformly faster online backend.
