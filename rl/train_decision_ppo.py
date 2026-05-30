"""
Decision-layer PPO training for DREAM.
=======================================

Trains a :class:`rl.policy.decision_policy.DecisionPolicy` on
:class:`rl.env.merger_decision_env.MergerDecisionEnv` using a compact,
dependency-free PPO implementation tailored for the 9-way discrete
action space.

Two modes:

* **Scratch**: start from a randomly initialised DecisionPolicy.
* **BC warm-start** (``--bc-checkpoint``): initialise from a
  behaviour-cloned checkpoint produced by :mod:`rl.train_bc` — this is
  the intended pipeline for the BC → PPO fine-tuning workflow.

The output checkpoint is saved in the same format as
:func:`rl.train_bc.load_decision_policy` consumes, so the trained
policy can be loaded by ``uncertainty_merger_DREAM.py`` via
``--rl-policy-mode decision --rl-decision-checkpoint <path>`` without
further conversion.

Usage
-----
    python -m rl.train_decision_ppo \
        --bc-checkpoint rl/checkpoints/decision_policy_bc.pt \
        --out           rl/checkpoints/decision_policy_ppo.pt \
        --total-steps   50000 \
        --rollout-steps 1024

    # Evaluation only (no training):
    python -m rl.train_decision_ppo --eval-only \
        --bc-checkpoint rl/checkpoints/decision_policy_bc.pt \
        --eval-episodes 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from rl.policy.decision_policy import (
    DEC_OBS_DIM,
    DEC_N_ACTIONS,
    DecisionPolicy,
)
from rl.train_bc import EXPECTED_SCHEMA_VERSION, load_decision_policy
from rl.env.merger_decision_env import MergerDecisionEnv


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    total_steps: int = 200_000       # was 50k — need more exploration with higher entropy
    rollout_steps: int = 2048        # was 1024 — longer rollouts capture full LC sequences
    n_epochs: int = 4
    minibatch_size: int = 256

    gamma: float = 0.99
    gae_lambda: float = 0.95

    clip_eps: float = 0.2
    lr: float = 1e-4                 # was 3e-4 — slower LR prevents fast collapse
    lr_schedule: str = "cosine"      # "cosine" | "constant" — cosine anneals lr→0 over total_steps
    max_grad_norm: float = 0.5
    entropy_coef: float = 0.05       # was 0.01 — 5x higher to maintain exploration in 9-action space
    vf_coef: float = 1.0             # was 0.5 — doubled so the critic keeps up with the policy

    log_interval: int = 1            # rollouts between stdout lines
    save_interval_rollouts: int = 20


# ---------------------------------------------------------------------------
# Rollout storage
# ---------------------------------------------------------------------------

class RolloutBuffer:
    def __init__(self, size: int, obs_dim: int, device: str = "cpu"):
        self.obs      = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions  = np.zeros((size,), dtype=np.int64)
        self.logp     = np.zeros((size,), dtype=np.float32)
        self.values   = np.zeros((size,), dtype=np.float32)
        self.rewards  = np.zeros((size,), dtype=np.float32)
        self.dones    = np.zeros((size,), dtype=np.float32)
        self.returns  = np.zeros((size,), dtype=np.float32)
        self.advs     = np.zeros((size,), dtype=np.float32)
        self.size = size
        self.device = device
        self.ptr = 0

    def add(self, obs, action, logp, value, reward, done):
        i = self.ptr
        self.obs[i]     = obs
        self.actions[i] = int(action)
        self.logp[i]    = float(logp)
        self.values[i]  = float(value)
        self.rewards[i] = float(reward)
        self.dones[i]   = float(done)
        self.ptr += 1

    def full(self) -> bool:
        return self.ptr >= self.size

    def compute_gae(self, last_value: float, gamma: float, lam: float):
        gae = 0.0
        for t in reversed(range(self.ptr)):
            next_val = last_value if t == self.ptr - 1 else self.values[t + 1]
            nonterm  = 1.0 - self.dones[t]
            delta    = self.rewards[t] + gamma * next_val * nonterm - self.values[t]
            gae      = delta + gamma * lam * nonterm * gae
            self.advs[t] = gae
        self.returns[: self.ptr] = self.advs[: self.ptr] + self.values[: self.ptr]

    def to_tensors(self, device: str):
        n = self.ptr
        return (
            torch.as_tensor(self.obs[:n],     device=device),
            torch.as_tensor(self.actions[:n], device=device),
            torch.as_tensor(self.logp[:n],    device=device),
            torch.as_tensor(self.values[:n],  device=device),
            torch.as_tensor(self.returns[:n], device=device),
            torch.as_tensor(self.advs[:n],    device=device),
        )

    def reset(self):
        self.ptr = 0


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def _build_policy_from_bc(bc_path: str | None, hidden: int,
                          device: str) -> DecisionPolicy:
    """Warm-start from a BC checkpoint, or cold-start a fresh policy."""
    if bc_path and os.path.isfile(bc_path):
        print(f"[ppo] warm-start from BC checkpoint: {bc_path}")
        return load_decision_policy(bc_path, device=device)
    print("[ppo] no BC checkpoint supplied — cold-starting policy")
    return DecisionPolicy(obs_dim=DEC_OBS_DIM,
                          n_actions=DEC_N_ACTIONS,
                          hidden=hidden).to(device)


def _eval_policy(policy: DecisionPolicy, env: MergerDecisionEnv,
                 n_episodes: int = 5, deterministic: bool = True) -> dict:
    """Roll out n_episodes and return aggregate metrics."""
    returns, lengths, lc_counts = [], [], []
    term_reasons: dict = {}
    entropy_sum = 0.0
    entropy_n = 0
    for _ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total = 0.0
        steps = 0
        lcs = 0
        prev_lane = int(env._inner._current_lane)
        while not done:
            with torch.no_grad():
                x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                logits, _v = policy(x)
                probs = F.softmax(logits, dim=-1)
                ent = -(probs * probs.clamp_min(1e-8).log()).sum(-1).mean().item()
                entropy_sum += ent
                entropy_n += 1
                if deterministic:
                    a = int(logits.argmax(-1).item())
                else:
                    a = int(torch.multinomial(probs, 1).item())
            obs, r, term, trunc, info = env.step(a)
            total += r
            steps += 1
            cur_lane = int(env._inner._current_lane)
            if cur_lane != prev_lane:
                lcs += 1
                prev_lane = cur_lane
            done = term or trunc
        returns.append(total)
        lengths.append(steps)
        lc_counts.append(lcs)
        reason = info.get("term_reason") or ("timeout" if trunc else "unknown")
        term_reasons[reason] = term_reasons.get(reason, 0) + 1
    return {
        "ep_return_mean":  float(np.mean(returns)),
        "ep_return_std":   float(np.std(returns)),
        "ep_length_mean":  float(np.mean(lengths)),
        "lc_count_mean":   float(np.mean(lc_counts)),
        "entropy_mean":    entropy_sum / max(1, entropy_n),
        "term_reasons":    term_reasons,
        "n_episodes":      int(n_episodes),
    }


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def _ppo_update(policy: DecisionPolicy, opt: torch.optim.Optimizer,
                buf: RolloutBuffer, cfg: PPOConfig, device: str) -> dict:
    obs_t, act_t, logp_old_t, val_old_t, ret_t, adv_t = buf.to_tensors(device)
    # Advantage normalization
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    # Return normalization — keep v_loss on a reasonable scale independent
    # of the absolute reward magnitude.  The critic learns normalized returns;
    # advantages (already normalized) are invariant to this.
    ret_mean = ret_t.mean()
    ret_std  = ret_t.std().clamp_min(1e-6)
    ret_norm = (ret_t - ret_mean) / ret_std

    n = obs_t.shape[0]
    idx = np.arange(n)

    losses_pi, losses_v, losses_ent = [], [], []
    for _epoch in range(cfg.n_epochs):
        np.random.shuffle(idx)
        for start in range(0, n, cfg.minibatch_size):
            mb = idx[start : start + cfg.minibatch_size]
            mb_t = torch.as_tensor(mb, device=device, dtype=torch.long)

            logp_new, v_new, logits = policy.log_prob(
                obs_t.index_select(0, mb_t), act_t.index_select(0, mb_t),
            )
            ratio = torch.exp(logp_new - logp_old_t.index_select(0, mb_t))
            adv_b = adv_t.index_select(0, mb_t)
            unclipped = ratio * adv_b
            clipped   = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_b
            pi_loss = -torch.minimum(unclipped, clipped).mean()

            # Clipped value loss (PPO2 style) on normalized returns
            ret_mb      = ret_norm.index_select(0, mb_t)
            val_old_mb  = val_old_t.index_select(0, mb_t)
            val_old_n   = (val_old_mb - ret_mean) / ret_std
            v_pred_n    = (v_new - ret_mean) / ret_std
            v_clipped_n = val_old_n + torch.clamp(
                v_pred_n - val_old_n, -cfg.clip_eps, cfg.clip_eps)
            v_loss_un = (v_pred_n    - ret_mb) ** 2
            v_loss_cl = (v_clipped_n - ret_mb) ** 2
            v_loss    = 0.5 * torch.maximum(v_loss_un, v_loss_cl).mean()

            probs   = F.softmax(logits, dim=-1)
            ent     = -(probs * probs.clamp_min(1e-8).log()).sum(-1).mean()
            ent_loss = -cfg.entropy_coef * ent

            loss = pi_loss + cfg.vf_coef * v_loss + ent_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            opt.step()

            losses_pi.append(float(pi_loss.item()))
            losses_v.append(float(v_loss.item()))
            losses_ent.append(float(ent.item()))

    return {
        "pi_loss":  float(np.mean(losses_pi)),
        "v_loss":   float(np.mean(losses_v)),
        "entropy":  float(np.mean(losses_ent)),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_decision_ppo(
    out_path: str,
    bc_checkpoint: str | None,
    cfg: PPOConfig,
    scenario: str = "random",
    warmup: bool = False,
    hidden: int = 128,
    device: str | None = None,
    log_path: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    env = MergerDecisionEnv(scenario=scenario, warmup=warmup)
    policy = _build_policy_from_bc(bc_checkpoint, hidden=hidden, device=device)
    policy.train()
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    _lr0 = cfg.lr   # base LR for cosine schedule

    buf = RolloutBuffer(cfg.rollout_steps, DEC_OBS_DIM, device=device)

    obs, _info = env.reset()
    total_steps = 0
    rollout_idx = 0
    ep_returns: list = []
    ep_lengths: list = []
    ep_return = 0.0
    ep_length = 0
    rollout_lc_actions = 0   # count LC actions (3-8) per rollout for diagnostics
    rollout_lc_rejected = 0  # count env-rejected LCs per rollout
    rollout_rterm_sums: dict = {}   # per-rollout sum of each reward term
    rollout_term_counts: dict = {   # per-rollout termination reason counts
        'collision': 0, 'offroad': 0, 'stall': 0, 'timeout': 0, 'none': 0,
    }

    log_rows: list = []
    t_start = time.time()

    while total_steps < cfg.total_steps:
        buf.reset()
        rollout_lc_actions = 0
        rollout_lc_rejected = 0
        rollout_rterm_sums = {}
        rollout_term_counts = {'collision': 0, 'offroad': 0, 'stall': 0,
                               'timeout': 0, 'none': 0}
        rollout_ep_lengths: list = []
        while not buf.full():
            with torch.no_grad():
                x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits, v = policy(x)
                dist = Categorical(logits=logits)
                a = int(dist.sample().item())
                logp = float(dist.log_prob(torch.tensor([a], device=device)).item())
                value = float(v.item())

            next_obs, r, term, trunc, _info = env.step(a)
            done = term or trunc
            buf.add(obs, a, logp, value, r, float(term))

            # Track LC statistics
            if a >= 3:
                rollout_lc_actions += 1
            if _info.get('lc_rejected', False):
                rollout_lc_rejected += 1

            # Aggregate per-term reward breakdown
            for k, v in (_info.get('r_terms') or {}).items():
                rollout_rterm_sums[k] = rollout_rterm_sums.get(k, 0.0) + float(v)

            obs = next_obs
            ep_return += r
            ep_length += 1
            total_steps += 1

            if done:
                reason = _info.get('term_reason') or 'none'
                if reason in rollout_term_counts:
                    rollout_term_counts[reason] += 1
                else:
                    rollout_term_counts['none'] += 1
                ep_returns.append(ep_return)
                ep_lengths.append(ep_length)
                rollout_ep_lengths.append(ep_length)
                ep_return = 0.0
                ep_length = 0
                obs, _info = env.reset()

            if buf.full() or total_steps >= cfg.total_steps:
                break

        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            _logits, v_last = policy(x)
            last_value = float(v_last.item())
        buf.compute_gae(last_value, cfg.gamma, cfg.gae_lambda)

        # --- Cosine LR schedule: lr decays from _lr0 to ~0 over total_steps ---
        if cfg.lr_schedule == "cosine":
            progress = min(1.0, total_steps / max(1, cfg.total_steps))
            lr_now = _lr0 * 0.5 * (1.0 + float(np.cos(np.pi * progress)))
            for pg in opt.param_groups:
                pg['lr'] = lr_now

        stats = _ppo_update(policy, opt, buf, cfg, device)

        # --- Adaptive entropy floor: if entropy drops too low, bump coeff ---
        _ENT_FLOOR = 0.3     # ~uniform over 3 modes; well above single-action collapse
        _ENT_CEIL  = 0.10    # restore original coeff when healthy
        if stats["entropy"] < _ENT_FLOOR and cfg.entropy_coef < 0.15:
            cfg.entropy_coef = min(cfg.entropy_coef * 1.5, 0.15)
        elif stats["entropy"] > 1.0 and cfg.entropy_coef > _ENT_CEIL:
            cfg.entropy_coef = max(cfg.entropy_coef * 0.9, 0.05)

        rollout_idx += 1
        recent = ep_returns[-10:] if ep_returns else [0.0]
        mean_ret = float(np.mean(recent))
        dt = time.time() - t_start
        lc_frac = rollout_lc_actions / max(1, cfg.rollout_steps)
        # Per-step training reward averaged over the rollout — this is the
        # curve that rises from low values and converges as the policy learns.
        reward_mean = float(np.mean(buf.rewards[: buf.ptr])) if buf.ptr > 0 else 0.0
        reward_sum  = float(np.sum(buf.rewards[: buf.ptr]))
        # Normalise per-term reward by rollout steps for comparability
        rterm_means = {f"r_{k}_mean": v / max(1, cfg.rollout_steps)
                       for k, v in rollout_rterm_sums.items()}
        term_fracs = {f"end_{k}": c / max(1, sum(rollout_term_counts.values()))
                      for k, c in rollout_term_counts.items()}
        ep_len_mean = float(np.mean(rollout_ep_lengths)) \
            if rollout_ep_lengths else float('nan')
        row = {
            "rollout":       rollout_idx,
            "steps":         total_steps,
            "ep_return_mean": mean_ret,
            "ep_length_mean": ep_len_mean,
            "reward_mean":   reward_mean,    # avg per-step reward in rollout
            "reward_sum":    reward_sum,     # total reward collected in rollout
            "pi_loss":       stats["pi_loss"],
            "v_loss":        stats["v_loss"],
            "entropy":       stats["entropy"],
            "entropy_coef":  cfg.entropy_coef,
            "lr":            float(opt.param_groups[0]['lr']),
            "lc_actions":    rollout_lc_actions,
            "lc_rejected":   rollout_lc_rejected,
            "lc_frac":       lc_frac,
            "elapsed_s":     dt,
            **rterm_means,
            **term_fracs,
        }
        log_rows.append(row)
        if rollout_idx % cfg.log_interval == 0:
            print(f"[ppo] rollout={rollout_idx:>3d}  steps={total_steps:>6d}  "
                  f"r_step={reward_mean:+.3f}  ep_ret(last10)={mean_ret:+.2f}  "
                  f"pi={stats['pi_loss']:+.3f}  v={stats['v_loss']:.3f}  "
                  f"ent={stats['entropy']:.3f}  ent_c={cfg.entropy_coef:.4f}  "
                  f"t={dt:.1f}s")

        if rollout_idx % cfg.save_interval_rollouts == 0:
            _save_checkpoint(policy, out_path, bc_checkpoint, hidden,
                             extra={"rollout": rollout_idx,
                                    "steps":   total_steps})

    _save_checkpoint(policy, out_path, bc_checkpoint, hidden,
                     extra={"rollout": rollout_idx, "steps": total_steps,
                            "final": True})
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(log_rows, f, indent=2)
        print(f"[ppo] wrote training log → {log_path}")

    env.close()
    return policy


def _save_checkpoint(policy: DecisionPolicy, out_path: str,
                     bc_source: str | None, hidden: int, extra: dict):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ckpt = {
        "state_dict":     {k: v.detach().cpu().clone()
                           for k, v in policy.state_dict().items()},
        "obs_dim":        DEC_OBS_DIM,
        "n_actions":      DEC_N_ACTIONS,
        "hidden":         hidden,
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "bc_source":      bc_source,
    }
    ckpt.update(extra or {})
    torch.save(ckpt, out_path)
    print(f"[ppo] saved checkpoint → {out_path}  ({extra})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out",           default="rl/checkpoints/decision_policy_ppo.pt")
    p.add_argument("--bc-checkpoint", default=None,
                   help="Warm-start from this BC checkpoint")
    p.add_argument("--scenario",      default="random")
    p.add_argument("--warmup",        action="store_true",
                   help="Warm up the DRIFT field on every reset (slow, off by default)")
    p.add_argument("--total-steps",   type=int, default=200_000)
    p.add_argument("--rollout-steps", type=int, default=2048)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--n-epochs",      type=int, default=4)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--entropy-coef",  type=float, default=0.05)
    p.add_argument("--hidden",        type=int, default=128)
    p.add_argument("--device",        default=None)
    p.add_argument("--log-path",      default=None)
    p.add_argument("--eval-only",     action="store_true")
    p.add_argument("--eval-episodes", type=int, default=5)
    return p.parse_args()


def main():
    args = _parse_args()

    if args.eval_only:
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        env = MergerDecisionEnv(scenario=args.scenario, warmup=args.warmup)
        policy = _build_policy_from_bc(args.bc_checkpoint, hidden=args.hidden,
                                       device=device)
        policy.eval()
        metrics = _eval_policy(policy, env, n_episodes=args.eval_episodes)
        print("[eval]", json.dumps(metrics, indent=2))
        env.close()
        return

    cfg = PPOConfig(
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        minibatch_size=args.minibatch_size,
        n_epochs=args.n_epochs,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
    )
    train_decision_ppo(
        out_path=args.out,
        bc_checkpoint=args.bc_checkpoint,
        cfg=cfg,
        scenario=args.scenario,
        warmup=args.warmup,
        hidden=args.hidden,
        device=args.device,
        log_path=args.log_path,
    )


if __name__ == "__main__":
    main()
