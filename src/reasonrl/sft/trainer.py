"""Supervised fine-tuning with prompt-masked cross-entropy."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from reasonrl.config import ExperimentConfig, OptimConfig
from reasonrl.data.collators import SFTCollator
from reasonrl.logging_utils import MetricsLogger
from reasonrl.models.load import save_checkpoint


def build_optimizer(model: torch.nn.Module, cfg: OptimConfig) -> torch.optim.Optimizer:
    """AdamW; no weight decay on biases / norms / embeddings-like 1-D params."""
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))


def build_scheduler(opt: torch.optim.Optimizer, cfg: OptimConfig, total_steps: int):
    warm = max(0, cfg.warmup_steps)
    total = max(1, total_steps)

    def lr_lambda(step: int) -> float:
        if warm and step < warm:
            return (step + 1) / warm
        if cfg.scheduler == "constant":
            return 1.0
        progress = min(1.0, (step - warm) / max(1, total - warm))
        if cfg.scheduler == "linear":
            return max(0.0, 1.0 - progress)
        if cfg.scheduler == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError(cfg.scheduler)

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def causal_lm_loss(model: torch.nn.Module, batch: dict) -> tuple[torch.Tensor, int]:
    """Mean token NLL over label positions (labels == -100 are ignored)."""
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    shift_logits = logits[:, :-1].float()
    shift_labels = batch["labels"][:, 1:]
    n_tokens = int((shift_labels != -100).sum())
    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    ) / max(1, n_tokens)
    return loss, n_tokens


def _to(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def sft_step(
    model: torch.nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float = 1.0,
    scheduler: Any = None,
) -> dict:
    """One full optimizer step on ``batch`` (used by tests and the trainer)."""
    model.train()
    loss, n_tokens = causal_lm_loss(model, batch)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_grad_norm
    )
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return {"loss": float(loss.detach()), "grad_norm": float(grad_norm), "tokens": n_tokens}


class SFTTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        train_ds: Dataset,
        cfg: ExperimentConfig,
        logger: MetricsLogger,
        eval_ds: Dataset | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model, self.tok, self.cfg, self.log = model, tokenizer, cfg, logger
        self.device = torch.device(device)
        g = torch.Generator().manual_seed(cfg.seed)
        collate = SFTCollator(tokenizer.pad_token_id)
        self.train_dl = DataLoader(train_ds, batch_size=cfg.sft.batch_size, shuffle=True, generator=g, collate_fn=collate)
        self.eval_dl = (
            DataLoader(eval_ds, batch_size=cfg.sft.batch_size, shuffle=False, collate_fn=collate) if eval_ds else None
        )
        steps_per_epoch = math.ceil(len(self.train_dl) / cfg.sft.grad_accum)
        self.total_steps = cfg.sft.max_steps or steps_per_epoch * cfg.sft.epochs
        self.opt = build_optimizer(model, cfg.optim)
        self.sched = build_scheduler(self.opt, cfg.optim, self.total_steps)

    @torch.no_grad()
    def evaluate_loss(self) -> float:
        if self.eval_dl is None:
            return float("nan")
        self.model.eval()
        tot, n = 0.0, 0
        for batch in self.eval_dl:
            loss, k = causal_lm_loss(self.model, _to(batch, self.device))
            tot += float(loss) * k
            n += k
        self.model.train()
        return tot / max(1, n)

    def train(self) -> list[dict]:
        cfg = self.cfg.sft
        step, micro = 0, 0
        self.model.train()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        acc_loss, acc_tokens = 0.0, 0
        done = False
        for epoch in range(10**9 if cfg.max_steps else cfg.epochs):
            for batch in self.train_dl:
                loss, n_tok = causal_lm_loss(self.model, _to(batch, self.device))
                (loss / cfg.grad_accum).backward()
                acc_loss += float(loss.detach()) / cfg.grad_accum
                acc_tokens += n_tok
                micro += 1
                if micro % cfg.grad_accum:
                    continue
                gn = torch.nn.utils.clip_grad_norm_(trainable, self.cfg.optim.max_grad_norm)
                self.opt.step()
                self.sched.step()
                self.opt.zero_grad(set_to_none=True)
                step += 1
                metrics = {
                    "train/loss": acc_loss,
                    "train/grad_norm": float(gn),
                    "train/tokens": acc_tokens,
                    "train/lr": self.sched.get_last_lr()[0],
                    "epoch": epoch,
                }
                if cfg.eval_every and step % cfg.eval_every == 0:
                    metrics["eval/loss"] = self.evaluate_loss()
                self.log.log(step, metrics)
                acc_loss, acc_tokens = 0.0, 0
                if step >= self.total_steps:
                    done = True
                    break
            if done or (not cfg.max_steps and epoch + 1 >= cfg.epochs):
                break
        final = {"eval/loss": self.evaluate_loss()}
        self.log.log(step, final)
        return self.log.history

    def save(self, path: str | Path | None = None) -> Path:
        return save_checkpoint(self.model, self.tok, path or self.cfg.sft.save_dir)
