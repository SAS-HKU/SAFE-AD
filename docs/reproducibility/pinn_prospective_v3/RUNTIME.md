# PINN Runtime Interpretation

The release reports two timing scopes because a neural forward pass is not the
same quantity as an online field update.

| Scope | Backend | Device | Scene/context (ms) | Solve/forward (ms) | Transfer (ms) | Complete (ms) | p95 (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| Paired field core | Prospective PDE | CPU | 0.00 | 2.64 | 0.00 | 2.64 | 3.84 |
| Paired field core | Context PINN | CPU | 0.93 | 2.09 | 0.04 | 3.08 | 5.66 |
| Paired field core | Context PINN | CUDA | 0.77 | 1.06 | 0.19 | 2.04 | 3.34 |
| Online HighwayEnv | Prospective PDE | CPU | 2.31 | 2.75 | 0.00 | 5.21 | 8.35 |
| Online HighwayEnv | Context PINN | CPU | 3.15 | 2.55 | 0.10 | 5.98 | 8.27 |
| Online HighwayEnv | Context PINN | CUDA | 3.45 | 1.69 | 0.46 | 5.80 | 7.60 |

Values are means over batch-size-one merge queries; `p95` is the 95th
percentile of complete latency. The paired field-core rows start from cached
ego-local coefficients. Under that scope, CUDA PINN is 22.7% faster than the
CPU numerical solve (`2.04` versus `2.64` ms). The online rows additionally
include source/transport/diffusion construction, local reprojection,
nearest-agent context, and interpolation setup. CUDA reduces the complete PINN
path by 3.1% relative to PINN CPU in this concurrent-load run, but it remains
11.3% slower than the complete numerical backend (`5.80` versus `5.21` ms).

The measurements were collected while a separate MetaDrive training process
used 8-12% GPU utilization and roughly 4 GB VRAM. They are retained as a
reproducible diagnostic, not a final uncontended hardware claim. Re-run both
benchmark commands after training finishes before inserting latency values in
the manuscript. Model loading, rendering, simulator stepping, and RL policy
inference are excluded. Consequently, the defensible current conclusion is:

> CUDA accelerates the learned operator itself, but CPU scene conditioning and
> host-device transfers prevent an end-to-end field-backend speedup at batch
> size one. The accepted PINN is behaviorally interchangeable with the
> numerical teacher; additional on-device or asynchronous context construction
> is required for a latency claim.

Raw samples and hardware-load metadata are stored in the adjacent JSON/CSV
files. Do not compare the kernel row with a policy-level runtime column.
