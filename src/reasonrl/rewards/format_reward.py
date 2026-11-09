"""Format reward: does the completion follow ``<think>...</think><answer>...</answer>``?

Two modes are implemented on purpose:

* ``naive``  - regex "are the tags present?". This is what many first
  implementations do, and it is hackable: ``<think></think><answer></answer>``
  earns full format reward with zero content (see PAPER.md, Section 6).
* ``strict`` - tags must appear exactly once, in order, the answer must be
  non-empty (math) or contain a fenced code block (code), and there must be no
  trailing text after ``</answer>``. The think block must be non-trivial.
"""

from __future__ import annotations

import re

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n.*?```", re.DOTALL)
MIN_THINK_CHARS = 8


def naive_format_score(completion: str) -> float:
    return 1.0 if ("<answer>" in completion and "</answer>" in completion) else 0.0


def strict_format_score(completion: str, task: str = "math") -> tuple[float, dict]:
    info = {
        "n_think": len(_THINK_RE.findall(completion)),
        "n_answer": len(_ANSWER_RE.findall(completion)),
    }
    answers = _ANSWER_RE.findall(completion)
    thinks = _THINK_RE.findall(completion)
    if len(answers) != 1:
        info["reason"] = "answer_tag_count"
        return 0.0, info
    answer = answers[0].strip()
    if not answer:
        info["reason"] = "empty_answer"
        return 0.0, info
    if task == "code" and not _CODE_FENCE_RE.search(answer):
        info["reason"] = "no_code_block"
        return 0.0, info
    tail = completion.split("</answer>", 1)[1].strip()
    if len(tail) > 0:
        info["reason"] = "trailing_text"
        return 0.5, info
    if len(thinks) != 1 or len(thinks[0].strip()) < MIN_THINK_CHARS:
        info["reason"] = "missing_or_trivial_think"
        return 0.5, info
    if completion.find("<think>") > completion.find("<answer>"):
        info["reason"] = "wrong_order"
        return 0.5, info
    info["reason"] = "ok"
    return 1.0, info


def format_score(completion: str, task: str = "math", mode: str = "strict") -> tuple[float, dict]:
    if mode == "naive":
        return naive_format_score(completion), {"reason": "naive"}
    if mode == "strict":
        return strict_format_score(completion, task)
    raise ValueError(f"unknown format mode: {mode}")
