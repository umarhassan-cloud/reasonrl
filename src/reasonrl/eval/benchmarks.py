"""Held-out evaluation with a fixed protocol shared by every method.

Evaluation scores *correctness only* (never the shaped/format reward), with
the same prompts, decoding parameters, token budget and seed for every
checkpoint, so SFT / GRPO / DPO numbers are directly comparable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from reasonrl.data.dataset import Example, encode, render_prompt
from reasonrl.eval.metrics import bootstrap_ci, paired_bootstrap, pass_at_k
from reasonrl.rewards.composite import RewardFunction
from reasonrl.rl.rollout import sample_completions


@dataclass
class EvalResult:
    name: str
    n: int
    metrics: dict
    per_example: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@torch.no_grad()
def evaluate_policy(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[Example],
    reward_fn: RewardFunction,
    name: str = "model",
    n_samples: int = 1,
    k_values: tuple[int, ...] | list[int] = (1,),
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_new_tokens: int = 256,
    batch_size: int = 16,
    max_prompt_len: int = 384,
    use_chat_template: bool = False,
    keep_completions: int = 3,
    bootstrap_samples: int = 2000,
) -> EvalResult:
    if temperature <= 0 and n_samples > 1:
        raise ValueError("greedy decoding with n_samples > 1 yields identical samples; set temperature > 0")
    per_example = []
    flat_ex: list[Example] = [ex for ex in examples for _ in range(n_samples)]
    prompts = [encode(tokenizer, render_prompt(ex, tokenizer, use_chat_template))[-max_prompt_len:] for ex in flat_ex]
    texts: list[str] = []
    truncated: list[bool] = []
    for i in range(0, len(prompts), batch_size):
        gen = sample_completions(
            model,
            prompts[i : i + batch_size],
            tokenizer.pad_token_id,
            tokenizer.eos_token_id,
            max_new_tokens,
            temperature,
            top_p,
        )
        texts.extend(tokenizer.decode(c, skip_special_tokens=True) for c in gen.completion_ids)
        truncated.extend(gen.truncated.tolist())
    outs = reward_fn.score_batch(flat_ex, texts)
    for j, ex in enumerate(examples):
        sl = slice(j * n_samples, (j + 1) * n_samples)
        o = outs[sl]
        c = int(sum(x.correctness >= 1.0 for x in o))
        per_example.append(
            {
                "id": ex.id,
                "task": ex.task,
                "n": n_samples,
                "correct": c,
                "pass@1": c / n_samples,
                "format_valid": sum(x.format >= 1.0 for x in o) / n_samples,
                "flags": sorted({f for x in o for f in x.flags}),
                "samples": [
                    {"completion": t, "extracted": x.extracted, "correct": x.correctness}
                    for t, x in list(zip(texts[sl], o))[:keep_completions]
                ],
            }
        )
    metrics: dict[str, float] = {}
    for k in k_values:
        if k <= n_samples:
            metrics[f"pass@{k}"] = sum(pass_at_k(n_samples, p["correct"], k) for p in per_example) / max(1, len(per_example))
    mean, lo, hi = bootstrap_ci([p["pass@1"] for p in per_example], n_boot=bootstrap_samples)
    metrics.update(
        {
            "pass@1_ci_lo": lo,
            "pass@1_ci_hi": hi,
            "format_valid_rate": sum(p["format_valid"] for p in per_example) / max(1, len(per_example)),
            "hack_flag_rate": sum(bool(p["flags"]) for p in per_example) / max(1, len(per_example)),
            "truncated_rate": sum(truncated) / max(1, len(truncated)),
            "mean_completion_chars": sum(len(t) for t in texts) / max(1, len(texts)),
        }
    )
    for task in sorted({e.task for e in examples}):
        sub = [p for p in per_example if p["task"] == task]
        metrics[f"{task}/pass@1"] = sum(p["pass@1"] for p in sub) / max(1, len(sub))
    return EvalResult(name=name, n=len(examples), metrics=metrics, per_example=per_example)


def compare(results: dict[str, EvalResult], baseline: str, n_boot: int = 2000) -> list[dict]:
    """Paired bootstrap of pass@1 of every run against ``baseline`` on aligned items."""
    base = results[baseline]
    base_by_id = {p["id"]: p["pass@1"] for p in base.per_example}
    rows = []
    for name, res in results.items():
        ids = [p["id"] for p in res.per_example if p["id"] in base_by_id]
        a = [p["pass@1"] for p in res.per_example if p["id"] in base_by_id]
        b = [base_by_id[i] for i in ids]
        row = {"name": name, **{k: v for k, v in res.metrics.items()}}
        if name != baseline and a:
            row["vs_baseline"] = paired_bootstrap(a, b, n_boot=n_boot)
        rows.append(row)
    return rows


def markdown_table(rows: list[dict], cols: tuple[str, ...] = ("pass@1", "format_valid_rate", "hack_flag_rate")) -> str:
    head = "| run | " + " | ".join(cols) + " | delta vs baseline [95% CI] |"
    sep = "|" + "---|" * (len(cols) + 2)
    lines = [head, sep]
    for r in rows:
        vals = " | ".join(f"{r.get(c, float('nan')):.3f}" for c in cols)
        vb = r.get("vs_baseline")
        delta = f"{vb['delta']:+.3f} [{vb['lo']:+.3f}, {vb['hi']:+.3f}] p={vb['p_value']:.3f}" if vb else "-"
        lines.append(f"| {r['name']} | {vals} | {delta} |")
    return "\n".join(lines)
