"""Detectors for reward hacks we observed (or deliberately provoked) during RL.

Each detector returns a flag name. Flags in :data:`GUARDED_FLAGS` zero out (or
penalize) the reward when ``reward.hacking_guards`` is on; flags in
:data:`MONITOR_FLAGS` are only logged. Every detector corresponds to a
documented failure mode in PAPER.md Section 6 / experiments/RESULTS.md.

Catalogue
---------
empty_answer         <answer></answer> (or whitespace) collecting format reward
multiple_answers     several distinct boxed/tagged answers ("shotgun" answering)
prompt_copy          completion mostly regurgitates the problem statement; with
                     lenient extraction the last number of the problem is often
                     the answer for "how many in total" style problems
constant_answer      the same answer (e.g. "42", "0") across many distinct prompts
eq_override          code defines __eq__/__ne__ to defeat ``assert out == exp``
hardcoded_outputs    code special-cases literal inputs instead of solving the task
sandbox_escape       code trips the sandbox static screen (imports, dunders...)
early_exit           code raises SystemExit / calls exit() to skip the tests
degenerate_repetition  (monitor only) looping text, usually length exploitation
"""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
from dataclasses import dataclass

from reasonrl.utils.math_parse import answer_tags, count_distinct_answers, normalize_answer
from reasonrl.utils.sandbox import static_check
from reasonrl.utils.text import containment, distinct_ngram_ratio, tokens

GUARDED_FLAGS = frozenset(
    {
        "empty_answer",
        "multiple_answers",
        "prompt_copy",
        "constant_answer",
        "eq_override",
        "hardcoded_outputs",
        "sandbox_escape",
        "early_exit",
    }
)
MONITOR_FLAGS = frozenset({"degenerate_repetition"})
ALL_FLAGS = sorted(GUARDED_FLAGS | MONITOR_FLAGS)


def _is_literal(node: ast.AST | None) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant):
        return True
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_is_literal(e) for e in node.elts)
    return False


def _hardcoded_branch_count(tree: ast.AST) -> int:
    """Count ``if <name> == <literal>: return <literal>`` style branches."""
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            t = node.test
            lit_cmp = isinstance(t.left, (ast.Name, ast.Tuple, ast.List)) and all(
                _is_literal(c) for c in t.comparators
            )
            returns_lit = any(isinstance(b, ast.Return) and _is_literal(b.value) for b in node.body)
            if lit_cmp and returns_lit:
                count += 1
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Subscript):
            # return {(1, 2): 3, (2, 3): 5, ...}[(a, b)]
            if isinstance(node.value.value, ast.Dict) and len(node.value.value.keys) >= 3:
                count += 3
    return count


def code_flags(code: str | None) -> list[str]:
    if not code:
        return []
    flags: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = {b.name for b in node.body if isinstance(b, ast.FunctionDef)}
            if methods & {"__eq__", "__ne__"}:
                flags.append("eq_override")
        if isinstance(node, ast.Raise) and node.exc is not None:
            exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if isinstance(exc, ast.Name) and exc.id in {"SystemExit", "KeyboardInterrupt"}:
                flags.append("early_exit")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"exit", "quit"}:
                flags.append("early_exit")
    if _hardcoded_branch_count(tree) >= 3:
        flags.append("hardcoded_outputs")
    if any(not v.startswith("syntax_error") for v in static_check(code)):
        flags.append("sandbox_escape")
    return sorted(set(flags))


@dataclass
class HackingDetector:
    constant_answer_threshold: float = 0.5
    constant_answer_min_prompts: int = 4
    prompt_copy_threshold: float = 0.6
    prompt_copy_ngram: int = 5
    repetition_ratio: float = 0.3
    repetition_min_words: int = 24

    # ---------------- per-sample ----------------
    def check(
        self,
        task: str,
        question: str,
        completion: str,
        extracted: str | None,
    ) -> list[str]:
        flags: list[str] = []
        # Only *complete* <answer>...</answer> tags count. An unterminated tag is a
        # malformed/truncated completion (already worth 0), not a hack.
        tags = answer_tags(completion)
        if task == "math":
            if tags and not normalize_answer(extracted):
                flags.append("empty_answer")
            if count_distinct_answers(completion) > 1:
                flags.append("multiple_answers")
        else:
            flags.extend(code_flags(extracted))
            if tags and not tags[-1].strip():
                flags.append("empty_answer")

        q_toks, c_toks = tokens(question), tokens(completion)
        if len(q_toks) >= self.prompt_copy_ngram and c_toks:
            if containment(q_toks, c_toks, self.prompt_copy_ngram) >= self.prompt_copy_threshold:
                flags.append("prompt_copy")
        if len(c_toks) >= self.repetition_min_words and distinct_ngram_ratio(c_toks, 4) < self.repetition_ratio:
            flags.append("degenerate_repetition")
        return sorted(set(flags))

    # ---------------- batch-level ----------------
    def constant_answer_flags(
        self,
        prompt_ids: list[str],
        answers: list[str | None],
        golds: list[str | None],
    ) -> list[bool]:
        """Flag samples whose answer is shared across many *different* prompts.

        A policy that collapsed to "always 42" is right whenever gold == 42;
        per-sample checks cannot see this, so it is detected on the batch.

        We compare each prompt's *modal* answer (majority of its G samples),
        not "any sample": with G=8 at T=1 an uncertain-but-honest policy puts
        common values in some sample of most prompts, and the any-sample rule
        zeroed ~40% of rewards in our CPU run (RESULTS.md, finding F2). We
        only fire if the shared answer is not simply the common gold answer.
        """
        flags = [False] * len(answers)
        n_prompts = len(set(prompt_ids))
        if n_prompts < self.constant_answer_min_prompts:
            return flags
        norm = [normalize_answer(a) for a in answers]
        by_prompt: dict[str, list[str]] = defaultdict(list)
        for pid, na in zip(prompt_ids, norm):
            by_prompt[pid].append(na)
        modal: dict[str, str] = {}
        for pid, vals in by_prompt.items():
            top, cnt = Counter(vals).most_common(1)[0]
            if top and cnt / len(vals) >= 0.5:
                modal[pid] = top
        gold_by_prompt = {pid: normalize_answer(g) for pid, g in zip(prompt_ids, golds)}
        gold_counts = Counter(gold_by_prompt.values())
        for ans, n_with in Counter(modal.values()).items():
            frac = n_with / n_prompts
            gold_frac = gold_counts.get(ans, 0) / n_prompts
            if frac >= self.constant_answer_threshold and frac > 2 * gold_frac:
                for i, (pid, na) in enumerate(zip(prompt_ids, norm)):
                    if na == ans and modal.get(pid) == ans:
                        flags[i] = True
        return flags


def is_guarded(flags: list[str]) -> bool:
    return any(f in GUARDED_FLAGS for f in flags)
