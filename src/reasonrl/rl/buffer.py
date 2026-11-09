"""Rollout storage for on-policy updates (one GRPO step = one buffer)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, fields

import torch


@dataclass
class RolloutBatch:
    """Token-level tensors are ``[B, T]``; logprob-aligned ones are ``[B, T-1]``.

    ``action_mask[:, j] == 1`` iff token ``j+1`` is a generated completion
    token (including EOS), i.e. it is an action taken by the policy.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    old_logprobs: torch.Tensor
    ref_logprobs: torch.Tensor | None
    advantages: torch.Tensor
    rewards: torch.Tensor

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def index(self, idx: torch.Tensor) -> RolloutBatch:
        kw = {}
        for f in fields(self):
            v = getattr(self, f.name)
            kw[f.name] = v[idx] if isinstance(v, torch.Tensor) else v
        return RolloutBatch(**kw)

    def to(self, device: torch.device | str) -> RolloutBatch:
        kw = {}
        for f in fields(self):
            v = getattr(self, f.name)
            kw[f.name] = v.to(device) if isinstance(v, torch.Tensor) else v
        return RolloutBatch(**kw)


class RolloutBuffer:
    def __init__(self, batch: RolloutBatch | None = None) -> None:
        self.batch = batch

    def set(self, batch: RolloutBatch) -> None:
        self.batch = batch

    def clear(self) -> None:
        self.batch = None

    def __len__(self) -> int:
        return 0 if self.batch is None else len(self.batch)

    def iter_minibatches(
        self, minibatch_size: int, shuffle: bool = True, generator: torch.Generator | None = None
    ) -> Iterator[RolloutBatch]:
        if self.batch is None:
            return
        n = len(self.batch)
        order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
        for start in range(0, n, minibatch_size):
            idx = order[start : start + minibatch_size]
            yield self.batch.index(idx.to(self.batch.input_ids.device))
