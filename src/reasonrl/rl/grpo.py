"""Group Relative Policy Optimization (GRPO).

Objective (Shao et al., 2024, DeepSeekMath; used for DeepSeek-R1-Zero / R1):

    J(theta) = E_{q, {o_i}~pi_old} [ 1/G sum_i 1/|o_i| sum_t
                 min( rho_{i,t} A_i, clip(rho_{i,t}, 1-eps, 1+eps) A_i )
                 - beta * KL_k3(pi_theta || pi_ref)_{i,t} ]

    rho_{i,t} = pi_theta(o_{i,t} | q, o_{i,<t}) / pi_old(o_{i,t} | q, o_{i,<t})
    A_i       = (r_i - mean(r)) / std(r)              (group-relative, no critic)
    KL_k3     = exp(ref - logp) - (ref - logp) - 1    (unbiased, always >= 0)

Supported variants (all config-switchable for ablations):
    loss_agg   seq_mean (original) | token_mean (DAPO) | constant (Dr. GRPO)
    adv_mode   std | mean_only (Dr. GRPO) | rloo
    clip       symmetric | clip-higher (DAPO, ``clip_eps_high``)
    kl_coef=0  removes the reference model entirely (DAPO / "no-KL" ablation)
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from reasonrl.config import ExperimentConfig
from reasonrl.data.dataset import Example, encode, render_prompt
from reasonrl.logging_utils import MetricsLogger
from reasonrl.models.load import save_checkpoint
from reasonrl.rewards.composite import RewardFunction
from reasonrl.rewards.hacking import ALL_FLAGS
from reasonrl.rl.advantages import group_relative_advantages, zero_std_groups
from reasonrl.rl.buffer import RolloutBatch, RolloutBuffer
from reasonrl.rl.rollout import masked_mean, sample_completions, token_logprobs
from reasonrl.sft.trainer import build_optimizer, build_scheduler

# --------------------------------------------------------------------------
# loss
# --------------------------------------------------------------------------


def kl_penalty(logp: torch.Tensor, ref_logp: torch.Tensor, estimator: str = "k3") -> torch.Tensor:
    """Per-token estimators of KL(pi || ref) from samples of pi (Schulman, 2020)."""
    log_r = ref_logp - logp
    if estimator == "k1":
        return -log_r
    if estimator == "k2":
        return 0.5 * log_r.pow(2)
    if estimator == "k3":
        return torch.exp(log_r) - log_r - 1.0
    raise ValueError(f"unknown KL estimator {estimator!r}")


def aggregate(per_token: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "seq_mean":
        per_seq = (per_token * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
        return per_seq.mean()
    if mode == "token_mean":
        return (per_token * mask).sum() / mask.sum().clamp(min=1.0)
    if mode == "constant":
        return (per_token * mask).sum() / (mask.size(0) * mask.size(1))
    raise ValueError(f"unknown loss aggregation {mode!r}")


def grpo_loss(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    ref_logp: torch.Tensor | None,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_eps: float = 0.2,
    clip_eps_high: float | None = None,
    kl_coef: float = 0.04,
    kl_estimator: str = "k3",
    loss_agg: str = "seq_mean",
) -> tuple[torch.Tensor, dict]:
    """All token tensors ``[B, T-1]``; ``advantages`` ``[B]``. Returns (loss, stats)."""
    hi = clip_eps if clip_eps_high is None else clip_eps_high
    ratio = torch.exp(logp - old_logp)
    adv = advantages[:, None].to(logp.dtype)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + hi) * adv
    pg = -torch.min(surr1, surr2)
    per_token = pg
    kl = None
    if ref_logp is not None:
        kl = kl_penalty(logp, ref_logp, kl_estimator)
        if kl_coef > 0:
            per_token = per_token + kl_coef * kl
    loss = aggregate(per_token, mask, loss_agg)
    with torch.no_grad():
        clipped = ((ratio < 1.0 - clip_eps) & (adv < 0)) | ((ratio > 1.0 + hi) & (adv > 0))
        stats = {
            "loss": float(loss.detach()),
            "pg_loss": float(aggregate(pg.detach(), mask, loss_agg)),
            "kl": float(masked_mean(kl.detach(), mask)) if kl is not None else 0.0,
            "clip_frac": float(masked_mean(clipped.float(), mask)),
            "ratio_mean": float(masked_mean(ratio.detach(), mask)),
        }
    return loss, stats


# --------------------------------------------------------------------------
# trainer
# --------------------------------------------------------------------------


@torch.no_grad()
def batched_logprobs(model: torch.nn.Module, ids: torch.Tensor, attn: torch.Tensor, chunk: int) -> torch.Tensor:
    outs = [token_logprobs(model, ids[i : i + chunk], attn[i : i + chunk]) for i in range(0, ids.size(0), chunk)]
    return torch.cat(outs, 0)


class GRPOTrainer:
    def __init__(
        self,
        policy: torch.nn.Module,
        ref_model: torch.nn.Module | None,
        tokenizer: Any,
        train_examples: list[Example],
        reward_fn: RewardFunction,
        cfg: ExperimentConfig,
        logger: MetricsLogger,
        eval_examples: list[Example] | None = None,
        device: torch.device | str = "cpu",
        eval_fn: Callable[[], dict] | None = None,
    ) -> None:
        if cfg.grpo.kl_coef > 0 and ref_model is None:
            raise ValueError("kl_coef > 0 requires a reference model")
        self.policy, self.ref, self.tok = policy, ref_model, tokenizer
        self.examples, self.eval_examples = train_examples, eval_examples
        self.reward_fn, self.cfg, self.g, self.log = reward_fn, cfg, cfg.grpo, logger
        self.device = torch.device(device)
        self.opt = build_optimizer(policy, cfg.optim)
        n_updates = self.g.steps * self.g.ppo_epochs * max(
            1, -(-self.g.prompts_per_step * self.g.group_size // self.g.minibatch_size)
        )
        self.sched = build_scheduler(self.opt, cfg.optim, n_updates)
        self.rng = random.Random(cfg.seed)
        self.torch_gen = torch.Generator().manual_seed(cfg.seed)
        self._order: list[int] = []
        self.buffer = RolloutBuffer()
        self.step_count = 0
        self.eval_fn = eval_fn
        self.samples_log: list[dict] = []

    # ---------------- data ----------------
    def _next_prompts(self) -> list[Example]:
        out = []
        while len(out) < self.g.prompts_per_step:
            if not self._order:
                self._order = list(range(len(self.examples)))
                self.rng.shuffle(self._order)
            out.append(self.examples[self._order.pop()])
        return out

    def _prompt_ids(self, ex: Example) -> list[int]:
        text = render_prompt(ex, self.tok, self.cfg.model.use_chat_template)
        return encode(self.tok, text)[-self.cfg.data.max_prompt_len :]

    # ---------------- rollout ----------------
    @torch.no_grad()
    def rollout(self, prompts: list[Example]) -> tuple[RolloutBatch, dict]:
        G = self.g.group_size
        rep = [ex for ex in prompts for _ in range(G)]  # group-major
        gen = sample_completions(
            self.policy,
            [self._prompt_ids(ex) for ex in rep],
            pad_id=self.tok.pad_token_id,
            eos_id=self.tok.eos_token_id,
            max_new_tokens=self.g.max_new_tokens,
            temperature=self.g.temperature,
            top_p=self.g.top_p,
        )
        texts = [self.tok.decode(c, skip_special_tokens=True) for c in gen.completion_ids]
        outs = self.reward_fn.score_batch(rep, texts)
        rewards = torch.tensor([o.total for o in outs], dtype=torch.float32)
        adv = group_relative_advantages(rewards, G, mode=self.g.adv_mode)
        zstd = zero_std_groups(rewards, G)

        chunk = max(1, self.g.minibatch_size)
        old_lp = batched_logprobs(self.policy, gen.input_ids, gen.attention_mask, chunk)
        ref_lp = (
            batched_logprobs(self.ref, gen.input_ids, gen.attention_mask, chunk) if self.ref is not None else None
        )
        batch = RolloutBatch(
            input_ids=gen.input_ids,
            attention_mask=gen.attention_mask,
            action_mask=gen.action_mask,
            old_logprobs=old_lp,
            ref_logprobs=ref_lp,
            advantages=adv.to(self.device),
            rewards=rewards.to(self.device),
        )
        corr = torch.tensor([o.correctness for o in outs])
        lengths = torch.tensor([len(c) for c in gen.completion_ids], dtype=torch.float32)
        info: dict[str, float] = {
            "reward/mean": float(rewards.mean()),
            "reward/std": float(rewards.std()) if rewards.numel() > 1 else 0.0,
            "reward/correct": float(corr.mean()),
            "reward/format": float(torch.tensor([o.format for o in outs]).mean()),
            "reward/format_valid_rate": float(torch.tensor([o.format >= 1.0 for o in outs]).float().mean()),
            "reward/guarded_frac": float(torch.tensor([o.guarded for o in outs]).float().mean()),
            "reward/pass_at_group": float(corr.view(-1, G).max(1).values.mean()),
            "groups/zero_std_frac": float(zstd.float().view(-1, G)[:, 0].mean()),
            "completion/len_mean": float(lengths.mean()),
            "completion/truncated_frac": float(gen.truncated.float().mean()),
        }
        for flag in ALL_FLAGS:
            info[f"hack/{flag}"] = float(sum(flag in o.flags for o in outs) / len(outs))
        self._last_samples = [
            {"id": ex.id, "completion": t, "reward": o.total, "correct": o.correctness, "flags": o.flags}
            for ex, t, o in zip(rep[:: max(1, G)], texts[:: max(1, G)], outs[:: max(1, G)])
        ]
        if self.g.filter_zero_std_groups:
            keep = (~zstd).nonzero(as_tuple=True)[0]
            batch = batch.index(keep.to(self.device))
            info["groups/kept_samples"] = float(len(keep))
        return batch, info

    # ---------------- update ----------------
    def update(self, batch: RolloutBatch) -> dict:
        self.policy.train()
        self.buffer.set(batch)
        stats_acc: dict[str, list[float]] = {}
        trainable = [p for p in self.policy.parameters() if p.requires_grad]
        for _ in range(self.g.ppo_epochs):
            for mb in self.buffer.iter_minibatches(self.g.minibatch_size, generator=self.torch_gen):
                logp = token_logprobs(self.policy, mb.input_ids, mb.attention_mask)
                loss, stats = grpo_loss(
                    logp,
                    mb.old_logprobs,
                    mb.ref_logprobs,
                    mb.advantages,
                    mb.action_mask,
                    clip_eps=self.g.clip_eps,
                    clip_eps_high=self.g.clip_eps_high,
                    kl_coef=self.g.kl_coef,
                    kl_estimator=self.g.kl_estimator,
                    loss_agg=self.g.loss_agg,
                )
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(trainable, self.cfg.optim.max_grad_norm)
                self.opt.step()
                self.sched.step()
                stats["grad_norm"] = float(gn)
                for k, v in stats.items():
                    stats_acc.setdefault(k, []).append(v)
        self.buffer.clear()
        # eval mode for the next rollout (dropout off); gradients still flow in update()
        self.policy.eval()
        return {f"train/{k}": sum(v) / len(v) for k, v in stats_acc.items()}

    def step(self) -> dict:
        t0 = time.time()
        batch, info = self.rollout(self._next_prompts())
        t1 = time.time()
        upd = self.update(batch) if len(batch) > 0 else {"train/skipped": 1.0}
        self.step_count += 1
        metrics = {
            **info,
            **upd,
            "train/lr": self.sched.get_last_lr()[0],
            "time/rollout_s": t1 - t0,
            "time/update_s": time.time() - t1,
        }
        self.log.log(self.step_count, metrics)
        return metrics

    def train(self) -> list[dict]:
        self.policy.eval()
        for _ in range(self.g.steps):
            self.step()
            s = self.step_count
            if self.eval_fn and self.g.eval_every and s % self.g.eval_every == 0:
                self.log.log(s, self.eval_fn())
            if self.g.save_every and s % self.g.save_every == 0:
                self.save(Path(self.g.save_dir) / f"step_{s}")
            self.samples_log.extend({"step": s, **x} for x in self._last_samples)
        already = bool(self.g.eval_every) and self.step_count % self.g.eval_every == 0
        if self.eval_fn and not already:
            self.log.log(self.step_count, self.eval_fn())
        return self.log.history

    def save(self, path: str | Path | None = None) -> Path:
        return save_checkpoint(self.policy, self.tok, path or self.g.save_dir)
