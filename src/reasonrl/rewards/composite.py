"""Composite reward = correctness + format, with hacking guards.

    r = w_c * correct + w_f * format          (if no guarded flag fires)
    r = -hacking_penalty                       (if a guarded flag fires)

Correctness is binary and verifiable; format is a small shaping term. With
``require_format_for_correctness`` correctness only counts when the format is
fully valid (used by the reward-shaping ablation).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from reasonrl.config import RewardConfig
from reasonrl.data.dataset import Example
from reasonrl.rewards.code_reward import code_correctness
from reasonrl.rewards.format_reward import format_score
from reasonrl.rewards.hacking import HackingDetector, is_guarded
from reasonrl.rewards.math_reward import math_correctness


@dataclass
class RewardOutput:
    total: float
    correctness: float
    format: float
    extracted: str | None
    flags: list = field(default_factory=list)
    guarded: bool = False
    info: dict = field(default_factory=dict)


class RewardFunction:
    def __init__(self, cfg: RewardConfig | None = None) -> None:
        self.cfg = cfg or RewardConfig()
        self.detector = HackingDetector(
            constant_answer_threshold=self.cfg.constant_answer_threshold,
            prompt_copy_threshold=self.cfg.prompt_copy_threshold,
        )

    # -- correctness only (used by evaluation; never shaped) --
    def correctness(self, example: Example, completion: str) -> tuple[float, str | None, dict]:
        if example.task == "math":
            return math_correctness(
                completion, example.answer, strict=self.cfg.strict_extraction, tol=self.cfg.math_tolerance
            )
        if example.task == "code":
            return code_correctness(
                completion,
                example.tests or [],
                example.entry_point,
                timeout=self.cfg.code_timeout_s,
                partial_credit=self.cfg.code_partial_credit,
                strict=self.cfg.strict_extraction,
            )
        raise ValueError(f"unknown task {example.task!r}")

    def _score_one(self, example: Example, completion: str) -> RewardOutput:
        corr, extracted, info = self.correctness(example, completion)
        fmt, finfo = format_score(completion, example.task, self.cfg.format_mode)
        flags = self.detector.check(example.task, example.question, completion, extracted)
        info = {**info, "format_reason": finfo.get("reason")}
        return RewardOutput(total=0.0, correctness=corr, format=fmt, extracted=extracted, flags=flags, info=info)

    def _finalize(self, out: RewardOutput) -> RewardOutput:
        c = self.cfg
        corr = out.correctness
        if c.require_format_for_correctness and out.format < 1.0:
            corr = 0.0
        total = c.correctness_weight * corr + c.format_weight * out.format
        out.guarded = c.hacking_guards and is_guarded(out.flags)
        if out.guarded:
            total = -c.hacking_penalty
        out.total = float(total)
        return out

    def __call__(self, example: Example, completion: str) -> RewardOutput:
        return self._finalize(self._score_one(example, completion))

    def score_batch(self, examples: list[Example], completions: list[str]) -> list[RewardOutput]:
        """Score a rollout batch; adds batch-level (cross-prompt) hacking checks."""
        if len(examples) != len(completions):
            raise ValueError("examples and completions must align")
        workers = self.cfg.code_workers if any(e.task == "code" for e in examples) else 1
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                outs = list(pool.map(self._score_one, examples, completions))
        else:
            outs = [self._score_one(e, c) for e, c in zip(examples, completions)]
        math_idx = [i for i, e in enumerate(examples) if e.task == "math"]
        if math_idx:
            const = self.detector.constant_answer_flags(
                [examples[i].id for i in math_idx],
                [outs[i].extracted for i in math_idx],
                [examples[i].answer for i in math_idx],
            )
            for i, hit in zip(math_idx, const):
                if hit:
                    outs[i].flags = sorted(set(outs[i].flags) | {"constant_answer"})
        return [self._finalize(o) for o in outs]


def build_reward_fn(cfg: RewardConfig | None = None) -> RewardFunction:
    return RewardFunction(cfg)
