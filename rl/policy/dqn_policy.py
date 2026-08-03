"""
Dueling Double-DQN policy for the DREAM decision layer.

Architecture
------------
Shared trunk (identical to DecisionPolicy) → dueling heads:

    Q(s, a) = V(s) + A(s, a) − mean_a A(s, a)

Keeping the trunk identical to `DecisionPolicy` means the BC checkpoint
(`decision_policy_bc.pt`) can warm-start the DQN trunk: we copy trunk
weights directly and reinitialise the dueling heads.  The policy head
from BC is not reused (BC's logits semantics ≠ Q-values), but the trunk's
learned feature representation transfers.

API mirrors DecisionPolicy where possible so the downstream
`decision_inference.decide_and_setup()` path works unchanged — `act(obs)`
returns a discrete action index in `[0, DEC_N_ACTIONS)`.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

from rl.policy.decision_policy import DEC_OBS_DIM, DEC_N_ACTIONS


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

if _HAS_TORCH:

    class DuelingQNet(nn.Module):
        """Dueling Q-network with the same trunk layout as DecisionPolicy."""

        def __init__(self, obs_dim: int = DEC_OBS_DIM,
                     n_actions: int = DEC_N_ACTIONS,
                     hidden: int = 128):
            super().__init__()
            self.n_actions = int(n_actions)
            # Trunk layout *identical* to DecisionPolicy so BC weights copy in.
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
            )
            # Dueling heads
            self.v_head = nn.Sequential(
                nn.Linear(hidden, hidden // 2), nn.Tanh(),
                nn.Linear(hidden // 2, 1),
            )
            self.a_head = nn.Sequential(
                nn.Linear(hidden, hidden // 2), nn.Tanh(),
                nn.Linear(hidden // 2, n_actions),
            )

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            h = self.trunk(obs)
            v = self.v_head(h)                          # (B, 1)
            a = self.a_head(h)                          # (B, A)
            # Dueling recomposition with mean-subtraction for identifiability
            q = v + (a - a.mean(dim=-1, keepdim=True))
            return q

        # --- BC warm-start helper -----------------------------------------
        def copy_trunk_from_bc(self, bc_state_dict: dict) -> int:
            """Copy trunk weights from a DecisionPolicy BC checkpoint.

            Returns the number of tensors copied (for log output).
            """
            own = self.state_dict()
            copied = 0
            for k, v in bc_state_dict.items():
                if k.startswith("trunk.") and k in own and own[k].shape == v.shape:
                    own[k].copy_(v)
                    copied += 1
            self.load_state_dict(own)
            return copied

        # --- Inference API (mirrors DecisionPolicy.act) -------------------
        @torch.no_grad()
        def act(self, obs: np.ndarray, deterministic: bool = True,
                epsilon: float = 0.0) -> int:
            """Greedy / ε-greedy action selection."""
            self.eval()
            if (not deterministic) and epsilon > 0.0 and np.random.rand() < epsilon:
                return int(np.random.randint(self.n_actions))
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            q = self.forward(x)
            return int(q.argmax(dim=-1).item())


else:
    class DuelingQNet:  # type: ignore
        def __init__(self, *a, **kw):
            raise ImportError("PyTorch not available — install torch to use DuelingQNet")
