"""Example schema, prompt templates and torch datasets.

All stages (SFT, GRPO, DPO, eval) render prompts through :func:`render_prompt`
so a model is never evaluated on a template it was not trained with.
Prompt and completion are tokenized *separately* and concatenated, which keeps
token boundaries identical between generation (RL) and training.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

MATH_INSTRUCTIONS = (
    "Solve the problem. Reason step by step inside <think></think>, then give only the "
    "final answer inside <answer></answer>."
)
CODE_INSTRUCTIONS = (
    "Write a Python function that solves the task. Reason briefly inside <think></think>, "
    "then put the complete function in a ```python code block inside <answer></answer>."
)
PROMPT_TEMPLATE = "### Problem\n{question}\n\n### Instructions\n{instructions}\n\n### Response\n"


@dataclass
class Example:
    id: str
    task: str  # "math" | "code"
    question: str
    answer: str | None = None  # gold final answer (math)
    solution: str | None = None  # full SFT target in the response format
    tests: list | None = None  # hidden unit tests (code)
    entry_point: str | None = None
    source: str = ""
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> Example:
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        extra = {k: v for k, v in d.items() if k not in cls.__dataclass_fields__}
        ex = cls(**known)
        ex.meta = {**ex.meta, **extra}
        return ex

    def to_dict(self) -> dict:
        return asdict(self)


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def write_jsonl(path: str | Path, rows: list[Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r.to_dict() if isinstance(r, Example) else r, ensure_ascii=False) + "\n")


def load_examples(paths: list[str] | str, limit: int | None = None) -> list[Example]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    out: list[Example] = []
    for p in paths:
        out.extend(Example.from_dict(r) for r in load_jsonl(p))
    ids = [e.id for e in out]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate example ids across data files")
    return out[:limit] if limit else out


def build_prompt_text(ex: Example) -> str:
    instructions = MATH_INSTRUCTIONS if ex.task == "math" else CODE_INSTRUCTIONS
    return PROMPT_TEMPLATE.format(question=ex.question.strip(), instructions=instructions)


def build_target_text(ex: Example) -> str:
    if ex.solution:
        return ex.solution
    if ex.answer is not None:
        return f"<think>The answer follows from the problem.</think>\n<answer>{ex.answer}</answer>"
    raise ValueError(f"example {ex.id} has neither solution nor answer")


def render_prompt(ex: Example, tokenizer: Any = None, use_chat_template: bool = False) -> str:
    text = build_prompt_text(ex)
    if use_chat_template and tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )
    return text


def encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


class SFTDataset(Dataset):
    """prompt + target + EOS; loss only on target tokens (labels=-100 on prompt)."""

    def __init__(
        self,
        examples: list[Example],
        tokenizer: Any,
        max_prompt_len: int = 384,
        max_target_len: int = 512,
        use_chat_template: bool = False,
    ) -> None:
        self.items = []
        eos = tokenizer.eos_token_id
        for ex in examples:
            p = encode(tokenizer, render_prompt(ex, tokenizer, use_chat_template))[-max_prompt_len:]
            t = encode(tokenizer, build_target_text(ex))[: max_target_len - 1] + [eos]
            self.items.append({"input_ids": p + t, "labels": [-100] * len(p) + t, "id": ex.id})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]


class PreferenceDataset(Dataset):
    """Rows of ``{"id", "prompt", "chosen", "rejected"}`` (raw text)."""

    def __init__(self, pairs: list[dict], tokenizer: Any, max_prompt_len: int = 384, max_target_len: int = 512):
        self.items = []
        eos = tokenizer.eos_token_id
        for row in pairs:
            p = encode(tokenizer, row["prompt"])[-max_prompt_len:]
            c = encode(tokenizer, row["chosen"])[: max_target_len - 1] + [eos]
            r = encode(tokenizer, row["rejected"])[: max_target_len - 1] + [eos]
            self.items.append({"prompt_ids": p, "chosen_ids": c, "rejected_ids": r, "id": row.get("id")})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]
