"""Critic-free advantage estimators for group sampling.

For a prompt q with G sampled completions o_1..o_G and scalar rewards r_i:

* ``std``       (GRPO, Shao et al. 2024):   A_i = (r_i - mean(r)) / (std(r) + eps)
* ``mean_only`` (Dr. GRPO, Liu et al. 2025): A_i =  r_i - mean(r)
  (removes the std normalization that up-weights near-deterministic prompts)
* ``rloo``      (Ahmadian et al. 2024):     A_i =  r_i - mean_{j != i}(r_j)

The advantage is a per-sequence scalar broadcast to every completion token
(outcome supervision).
"""

from __future__ import annotations

import torch


def group_relative_advantages(
    rewards: torch.Tensor,
    group_size: int,
    mode: str = "std",
    eps: float = 1e-4,
) -> torch.Tensor:
    """``rewards``: flat ``[N*G]`` ordered group-major (all G samples of prompt 0 first)."""
    if group_size < 2:
        raise ValueError("group-relative advantages need group_size >= 2")
    if rewards.numel() % group_size:
        raise ValueError(f"{rewards.numel()} rewards not divisible by group_size={group_size}")
    r = rewards.float().view(-1, group_size)
    mean = r.mean(dim=1, keepdim=True)
    if mode == "std":
        std = r.std(dim=1, keepdim=True)  # unbiased, as in the reference implementations
        adv = (r - mean) / (std + eps)
    elif mode == "mean_only":
        adv = r - mean
    elif mode == "rloo":
        adv = (r - mean) * group_size / (group_size - 1)  # == r_i - mean_{j!=i} r_j
    else:
        raise ValueError(f"unknown advantage mode {mode!r}")
    return adv.reshape(-1)


def zero_std_groups(rewards: torch.Tensor, group_size: int, atol: float = 1e-8) -> torch.Tensor:
    """Bool ``[N*G]``: True where the sample's group has identical rewards.

    Such groups produce zero advantage and contribute no policy gradient (only
    KL); DAPO's "dynamic sampling" drops them.
    """
    r = rewards.float().view(-1, group_size)
    same = (r.max(dim=1).values - r.min(dim=1).values) <= atol
    return same.repeat_interleave(group_size)


def broadcast_to_tokens(adv: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return adv[:, None] * mask
