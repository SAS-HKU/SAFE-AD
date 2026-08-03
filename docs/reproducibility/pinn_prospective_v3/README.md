# Prospective PINN v3 Reproducibility Evidence

This directory contains compact, machine-readable summaries from the accepted
checkpoint. Raw naturalistic trajectories, teacher caches, and generated
renderings are excluded because of dataset licensing and repository size.

- `construct_validity_*`: recording-level future-occupancy tests on held-out
  inD recordings 06-11. Future trajectories are labels only.
- `heldout_fidelity_summary.json`: numerical-to-PINN field and gradient
  agreement on recording-disjoint naturalistic and HighwayEnv scenes.
- `backend_swap_summary.json`: paired ten-seed closed-loop merge results for a
  fixed 33-D field-observing policy.
- `field_core_runtime_summary.*`: paired numerical/PINN latency after ego-local
  coefficient construction.
- `online_runtime_summary.*`: complete field-backend latency on a lane-valid
  HighwayEnv IDM merge trajectory.
- `RUNTIME.md` and `runtime_table.tex`: scope-safe interpretation and a compact
  manuscript table.

Regenerate every file with the commands in
[`../../pinn_multirecord_backend_commands.md`](../../pinn_multirecord_backend_commands.md).
The runtime summaries in this directory were collected while another CUDA
training job was active; rerun them under an idle GPU for final hardware claims.
