"""LoRA (Hu et al., 2021): native implementation + optional PEFT backend.

The native version supports ``nn.Linear`` and GPT-2's ``Conv1D`` (transposed
weight layout), so tests can exercise LoRA on the tiny model without PEFT.
Checkpoints are always saved merged, so downstream stages load plain models.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from reasonrl.config import LoRAConfig


def _is_conv1d(m: nn.Module) -> bool:
    return type(m).__name__ == "Conv1D" and hasattr(m, "nf")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Module, r: int, alpha: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        self.conv1d = _is_conv1d(base)
        w = base.weight
        in_f, out_f = (w.shape[0], w.shape[1]) if self.conv1d else (w.shape[1], w.shape[0])
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, in_f, dtype=w.dtype, device=w.device))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r, dtype=w.dtype, device=w.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return self.base(x) + delta * self.scaling

    def delta_weight(self) -> torch.Tensor:
        d = (self.lora_B @ self.lora_A) * self.scaling  # [out, in]
        return d.t() if self.conv1d else d

    @torch.no_grad()
    def merged(self) -> nn.Module:
        self.base.weight.add_(self.delta_weight().to(self.base.weight.dtype))
        for p in self.base.parameters():
            p.requires_grad_(True)
        return self.base


def _replace(model: nn.Module, cfg: LoRAConfig) -> int:
    n = 0
    targets = set(cfg.target_modules)
    for _name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in targets and (isinstance(child, nn.Linear) or _is_conv1d(child)):
                setattr(module, child_name, LoRALinear(child, cfg.r, cfg.alpha, cfg.dropout))
                n += 1
    return n


def apply_lora(model: nn.Module, cfg: LoRAConfig) -> nn.Module:
    if not cfg.enabled:
        return model
    if cfg.backend == "peft":
        from peft import LoraConfig as PeftLoraConfig
        from peft import get_peft_model

        present = {n.split(".")[-1] for n, _ in model.named_modules()}
        pcfg = PeftLoraConfig(
            r=cfg.r,
            lora_alpha=cfg.alpha,
            lora_dropout=cfg.dropout,
            target_modules=[t for t in cfg.target_modules if t in present],
            task_type="CAUSAL_LM",
        )
        return get_peft_model(model, pcfg)
    for p in model.parameters():
        p.requires_grad_(False)
    n = _replace(model, cfg)
    if n == 0:
        raise ValueError(f"LoRA: no modules matched {cfg.target_modules}")
    return model


def merge_lora(model: nn.Module) -> nn.Module:
    if hasattr(model, "merge_and_unload"):
        return model.merge_and_unload()
    for module in list(model.modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, child_name, child.merged())
    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
