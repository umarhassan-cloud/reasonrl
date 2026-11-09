"""Lexical helpers shared by contamination checks and hacking detectors."""

from __future__ import annotations

import re

_PUNCT_RE = re.compile(r"[^\w\s]")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def normalize_text(text: str, mask_numbers: bool = False) -> str:
    """Lowercase, drop punctuation, collapse whitespace; optionally mask numbers.

    ``mask_numbers=True`` catches *templated* near-duplicates where only the
    numbers differ ("Tom has 3 apples" vs "Tom has 5 apples").
    """
    s = _PUNCT_RE.sub(" ", text.lower())
    if mask_numbers:
        s = _NUM_RE.sub(" <num> ", s)
    return " ".join(s.split())


def tokens(text: str, mask_numbers: bool = False) -> list[str]:
    return normalize_text(text, mask_numbers).split()


def word_ngrams(toks: list[str], n: int) -> set[tuple[str, ...]]:
    if n <= 0:
        raise ValueError("n must be positive")
    if len(toks) < n:
        return set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def containment(source: list[str], target: list[str], n: int) -> float:
    """Fraction of ``source``'s n-grams that also occur in ``target``."""
    n_eff = min(n, len(source))
    if n_eff == 0:
        return 0.0
    src = word_ngrams(source, n_eff)
    if not src:
        return 0.0
    tgt = word_ngrams(target, n_eff)
    return len(src & tgt) / len(src)


def distinct_ngram_ratio(toks: list[str], n: int = 4) -> float:
    """Unique / total n-grams; low values indicate degenerate repetition."""
    total = len(toks) - n + 1
    if total <= 0:
        return 1.0
    return len(word_ngrams(toks, n)) / total
