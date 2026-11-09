"""Verifiable math reward: extract final answer, compare to gold."""

from __future__ import annotations

from reasonrl.utils.math_parse import extract_final_answer, is_equiv


def math_correctness(
    completion: str,
    gold: str | None,
    strict: bool = True,
    tol: float = 1e-6,
) -> tuple[float, str | None, dict]:
    """Return ``(score in {0,1}, extracted_answer, info)``."""
    pred, method = extract_final_answer(completion, strict=strict)
    correct = is_equiv(pred, gold, tol=tol)
    return float(correct), pred, {"extract_method": method}


class MathReward:
    """Callable wrapper so the reward can be used standalone."""

    def __init__(self, strict: bool = True, tol: float = 1e-6) -> None:
        self.strict = strict
        self.tol = tol

    def __call__(self, completion: str, gold: str) -> float:
        return math_correctness(completion, gold, self.strict, self.tol)[0]
