# Prospective-Field Merge PPO (20k)

This checkpoint is released only to reproduce the numerical-to-PINN backend
swap. It is a Stable-Baselines3 PPO policy trained for 20,000 steps in
`merge-v0` with continuous control, the A5 social reward, and eight prospective
field descriptors appended to the 25-D stock observation. The resulting model
expects 33 observations and two continuous actions.

- Training seed: 2026
- Training field backend: prospective numerical teacher
- Traffic preset: medium
- SHA-256: `53b028e49946c74af3696cecf9004b06a2d64f536e6cf6a9ce399dd7f59f2e91`

This short-run policy is an instrumentation checkpoint, not the headline RL
baseline. It exists so the fixed-policy action and closed-loop backend swap can
be reproduced without retraining.
