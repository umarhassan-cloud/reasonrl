"""Direct Preference Optimization (Rafailov et al., 2023).

    L_DPO = -E[ log sigmoid( beta * ( (log pi(y_w|x) - log ref(y_w|x))
                                     - (log pi(y_l|x) - log ref(y_l|x)) ) ) ]

Preference pairs come from the *same* verifiable reward used by GRPO:
we sample ``samples_per_prompt`` completions from the SFT model (matching
GRPO's group size), score them, and pair a correct (chosen) with an incorrect
(rejected) completion. This makes DPO the offline counterpart of GRPO with an
identical reward signal and sampling budget per prompt (see DESIGN.md).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from reasonrl.config import ExperimentConfig
from reasonrl.data.collators import PreferenceCollator
from reasonrl.data.dataset import Example, PreferenceDataset, encode, render_prompt
from reasonrl.logging_utils import MetricsLogger
from reasonrl.models.load import save_checkpoint
from reasonrl.rewards.composite import RewardFunction, RewardOutput
from reasonrl.rl.rollout import sample_completions, sequence_logps
from reasonrl.sft.trainer import build_optimizer, build_scheduler

# --------------------------------------------------------------------------
# loss
# --------------------------------------------------------------------------


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    loss_type: str = "sigmoid",
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """All inputs ``[B]`` summed sequence log-probs. Returns (loss, metrics).

    ``sigmoid``: standard DPO; ``label_smoothing`` > 0 gives conservative DPO
    (cDPO). ``ipo``: Azar et al. 2023 squared loss. ``hinge``: SLiC-style.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios
    if loss_type == "sigmoid":
        losses = (
            -F.logsigmoid(beta * logits) * (1 - label_smoothing)
            - F.logsigmoid(-beta * logits) * label_smoothing
        )
    elif loss_type == "ipo":
        losses = (logits - 1.0 / (2.0 * beta)) ** 2
    elif loss_type == "hinge":
        losses = torch.relu(1.0 - beta * logits)
    else:
        raise ValueError(f"unknown DPO loss type {loss_type!r}")
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    metrics = {
        "loss": float(losses.detach().mean()),
        "rewards/chosen": float(chosen_rewards.mean()),
        "rewards/rejected": float(rejected_rewards.mean()),
        "rewards/margin": float((chosen_rewards - rejected_rewards).mean()),
        "rewards/accuracy": float((chosen_rewards > rejected_rewards).float().mean()),
        "logps/chosen": float(policy_chosen_logps.detach().mean()),
        "logps/rejected": float(policy_rejected_logps.detach().mean()),
    }
    return losses.mean(), metrics


# --------------------------------------------------------------------------
# preference data
# --------------------------------------------------------------------------


def build_preference_pairs(
    examples: list[Example],
    completions: list[list[str]],
    scores: list[list[RewardOutput]],
    prompts: list[str],
    strategy: str = "correct_vs_incorrect",
    pairs_per_prompt: int = 2,
    gold_as_chosen: bool = False,
    seed: int = 0,
) -> list[dict]:
    """Turn scored samples into (chosen, rejected) pairs.

    * Hacked completions (guarded flags) are never used as ``chosen``; they
      are preferentially used as ``rejected`` so DPO learns to avoid them.
    * ``correct_vs_incorrect`` requires a correctness gap.
    * ``best_vs_worst`` uses the shaped total reward and requires margin > 0.
    * ``gold_as_chosen`` backfills a missing correct sample with the reference
      solution (off-policy; disabled by default for a fair GRPO comparison).
    """
    rng = random.Random(seed)
    pairs: list[dict] = []
    for ex, comps, outs, prompt in zip(examples, completions, scores, prompts):
        items = list(zip(comps, outs))
        if strategy == "correct_vs_incorrect":
            good = [c for c, o in items if o.correctness >= 1.0 and not o.guarded]
            bad_scored = [(not o.guarded, c) for c, o in items if o.correctness < 1.0 or o.guarded]
            bad = [c for _, c in sorted(bad_scored, key=lambda t: t[0])]  # hacked samples first
            if not good and gold_as_chosen and ex.solution:
                good = [ex.solution]
            good = list(dict.fromkeys(good))
            bad = list(dict.fromkeys(bad))
            rng.shuffle(good)
            for k in range(min(pairs_per_prompt, len(good) * len(bad))):
                pairs.append(
                    {"id": ex.id, "prompt": prompt, "chosen": good[k % len(good)], "rejected": bad[k % len(bad)]}
                )
        elif strategy == "best_vs_worst":
            ranked = sorted(items, key=lambda t: t[1].total, reverse=True)
            for k in range(min(pairs_per_prompt, len(ranked) // 2)):
                (c, oc), (r, orj) = ranked[k], ranked[-1 - k]
                if oc.total > orj.total and not oc.guarded and c != r:
                    pairs.append({"id": ex.id, "prompt": prompt, "chosen": c, "rejected": r})
        else:
            raise ValueError(f"unknown pair strategy {strategy!r}")
    return pairs


@torch.no_grad()
def generate_preference_pairs(
    policy: torch.nn.Module,
    tokenizer: Any,
    examples: list[Example],
    reward_fn: RewardFunction,
    cfg: ExperimentConfig,
    batch_prompts: int = 4,
) -> tuple[list[dict], dict]:
    d = cfg.dpo
    all_comps, all_scores, prompts = [], [], []
    for i in range(0, len(examples), batch_prompts):
        chunk = examples[i : i + batch_prompts]
        texts = [render_prompt(ex, tokenizer, cfg.model.use_chat_template) for ex in chunk]
        rep_ids = [encode(tokenizer, t)[-cfg.data.max_prompt_len :] for t in texts for _ in range(d.samples_per_prompt)]
        gen = sample_completions(
            policy, rep_ids, tokenizer.pad_token_id, tokenizer.eos_token_id, d.max_new_tokens, d.temperature, d.top_p
        )
        decoded = [tokenizer.decode(c, skip_special_tokens=True) for c in gen.completion_ids]
        rep_ex = [ex for ex in chunk for _ in range(d.samples_per_prompt)]
        outs = reward_fn.score_batch(rep_ex, decoded)
        for j in range(len(chunk)):
            sl = slice(j * d.samples_per_prompt, (j + 1) * d.samples_per_prompt)
            all_comps.append(decoded[sl])
            all_scores.append(outs[sl])
        prompts.extend(texts)
    pairs = build_preference_pairs(
        examples, all_comps, all_scores, prompts, d.pair_strategy, d.pairs_per_prompt, d.gold_as_chosen, cfg.seed
    )
    n = len(examples)
    stats = {
        "pairs": len(pairs),
        "prompts": n,
        "prompts_with_pair": len({p["id"] for p in pairs}),
        "sample_correct_rate": sum(o.correctness for s in all_scores for o in s) / max(1, n * d.samples_per_prompt),
    }
    return pairs, stats


# --------------------------------------------------------------------------
# trainer
# --------------------------------------------------------------------------


def _concat_logps(model: torch.nn.Module, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    lp = sequence_logps(model, batch["input_ids"], batch["attention_mask"], batch["labels"])
    n = batch["n_pairs"]
    return lp[:n], lp[n:]


class DPOTrainer:
    def __init__(
        self,
        policy: torch.nn.Module,
        ref_model: torch.nn.Module,
        tokenizer: Any,
        pairs: list[dict],
        cfg: ExperimentConfig,
        logger: MetricsLogger,
        device: torch.device | str = "cpu",
    ) -> None:
        if not pairs:
            raise ValueError("DPO needs at least one preference pair")
        self.policy, self.ref, self.tok, self.cfg, self.log = policy, ref_model, tokenizer, cfg, logger
        self.device = torch.device(device)
        ds = PreferenceDataset(pairs, tokenizer, cfg.data.max_prompt_len, cfg.data.max_target_len)
        self.collate = PreferenceCollator(tokenizer.pad_token_id)
        g = torch.Generator().manual_seed(cfg.seed)
        self.dl = DataLoader(
            _Indexed(ds), batch_size=cfg.dpo.batch_size, shuffle=True, generator=g, collate_fn=self._collate_with_idx
        )
        self.ds = ds
        self.ref_cache = self._precompute_reference(ds)
        total = cfg.dpo.epochs * -(-len(self.dl) // cfg.dpo.grad_accum)
        self.opt = build_optimizer(policy, cfg.optim)
        self.sched = build_scheduler(self.opt, cfg.optim, total)

    def _collate_with_idx(self, batch: list[tuple[int, dict]]) -> dict:
        idx = [i for i, _ in batch]
        out = self.collate([b for _, b in batch])
        out["idx"] = torch.tensor(idx)
        return out

    @torch.no_grad()
    def _precompute_reference(self, ds: PreferenceDataset) -> torch.Tensor:
        """Reference log-probs never change: compute once, saves a forward/step."""
        self.ref.eval()
        out = torch.zeros(len(ds), 2)
        bs = self.cfg.dpo.batch_size
        for s in range(0, len(ds), bs):
            items = [ds[i] for i in range(s, min(len(ds), s + bs))]
            b = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in self.collate(items).items()}
            c, r = _concat_logps(self.ref, b)
            out[s : s + len(items), 0] = c.float().cpu()
            out[s : s + len(items), 1] = r.float().cpu()
        return out

    def train(self) -> list[dict]:
        d = self.cfg.dpo
        self.policy.train()
        trainable = [p for p in self.policy.parameters() if p.requires_grad]
        step, micro = 0, 0
        acc: dict[str, float] = {}
        for epoch in range(d.epochs):
            for batch in self.dl:
                b = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                pc, pr = _concat_logps(self.policy, b)
                ref = self.ref_cache[batch["idx"]].to(self.device)
                loss, m = dpo_loss(pc, pr, ref[:, 0], ref[:, 1], d.beta, d.loss_type, d.label_smoothing)
                (loss / d.grad_accum).backward()
                for k, v in m.items():
                    acc[k] = acc.get(k, 0.0) + v / d.grad_accum
                micro += 1
                if micro % d.grad_accum:
                    continue
                gn = torch.nn.utils.clip_grad_norm_(trainable, self.cfg.optim.max_grad_norm)
                self.opt.step()
                self.sched.step()
                self.opt.zero_grad(set_to_none=True)
                step += 1
                self.log.log(
                    step,
                    {**{f"train/{k}": v for k, v in acc.items()}, "train/grad_norm": float(gn),
                     "train/lr": self.sched.get_last_lr()[0], "epoch": epoch},
                )
                acc = {}
        return self.log.history

    def save(self, path: str | Path | None = None) -> Path:
        return save_checkpoint(self.policy, self.tok, path or self.cfg.dpo.save_dir)


class _Indexed(torch.utils.data.Dataset):
    def __init__(self, ds: PreferenceDataset) -> None:
        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int) -> tuple[int, dict]:
        return i, self.ds[i]
