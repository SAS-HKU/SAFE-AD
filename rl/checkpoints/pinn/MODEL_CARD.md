# Prospective Context PINN v3

## Intended use

`pinn_prospective_context_v3_domain_conditioned.pt` approximates the causal
finite-horizon prospective risk teacher used by SAFE-AD. It accepts a
17-channel ego-local scene tensor and predicts the full propagated risk field
on a `41 x 141` grid covering `[-40, 100] m x [-20, 20] m`. It is intended for
field validation, HighwayEnv descriptor queries, and numerical-to-PINN backend
swap experiments. It is not a perception model or a standalone driving policy.

## Provenance

- Model family: `prospective_context_pinn`
- Teacher: `prospective_v2`, a 3 s discounted transport-diffusion integral
- Architecture: width 16, dilations `(1, 2, 4, 8, 16, 1)`
- Training: 6,000 AdamW steps, batch size 8, seed 2026
- Loss weights: field 1.0, gradient 0.30, physics residual 0.002
- Coordinate frame: ego-local; domain-conditioning channel enabled
- SHA-256: `58eca2885fdcc1bdd85ff8f05b04ed5e936ec517bd8323859f0656171af39948`

Calibration recordings are inD 01-05 and HighwayEnv seeds 0-2 for highway,
merge, intersection, and roundabout. Held-out evaluation uses inD 06-11 and
HighwayEnv seeds 100-102 for the same four scenario families. Naturalistic
datasets are not redistributed; obtain them from the original provider.

## Verified evidence

On held-out HighwayEnv frames, the field correlation with the numerical teacher
is 0.925. For merge frames, correlation is 0.977, active-gradient angular error
is 7.4 degrees, and equal-mass hotspot intersection-over-union is 0.854. In a
paired ten-seed merge backend swap, the mean action-vector difference is
`6.68e-5`; neither backend collides, and the PINN-minus-teacher progress change
is -0.061 m over approximately 586 m.

Runtime depends on scope. CUDA accelerates the neural kernel, but complete
online latency also includes scene projection and multiscale context creation.
The current release therefore claims behavioral interchangeability, not an
unqualified end-to-end speedup. Re-run the paired timing commands after other
GPU workloads have stopped before using latency values in a paper.

## Reproduction

See [`../../../docs/pinn_multirecord_backend_commands.md`](../../../docs/pinn_multirecord_backend_commands.md)
for cache construction, training, held-out validation, runtime decomposition,
backend swap, and qualitative rendering commands.
