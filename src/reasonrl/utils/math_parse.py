"""Final-answer extraction and equivalence checking for math rewards.

Design notes
------------
* Extraction is *strict* by default: only an ``<answer>...</answer>`` tag or a
  ``\\boxed{...}`` counts. The lenient "last number in the text" fallback is
  kept only for ablations because it is trivially hackable (a completion that
  copies the problem statement often ends with a number from the problem).
* Equivalence is checked in increasing cost order: normalized string match ->
  exact rational arithmetic -> float tolerance -> a *guarded* SymPy check.
  SymPy's parser calls ``eval`` internally, so model output is only handed to
  SymPy after it passes a strict character/identifier whitelist.
"""

from __future__ import annotations

import re
from fractions import Fraction

_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?|-?\.\d+")
_THOUSANDS_RE = re.compile(r"(?<![\d.])(\d{1,3}(?:,\d{3})+)(?![\d])")


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------


def all_boxed(text: str) -> list[str]:
    """Contents of every ``\\boxed{...}`` / ``\\fbox{...}`` with brace matching."""
    found: list[tuple[int, str]] = []
    for marker in ("\\boxed{", "\\fbox{"):
        start = 0
        while True:
            idx = text.find(marker, start)
            if idx < 0:
                break
            i = idx + len(marker)
            depth = 1
            j = i
            while j < len(text) and depth > 0:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                j += 1
            if depth == 0:
                found.append((idx, text[i : j - 1]))
            start = j
    found.sort(key=lambda t: t[0])
    return [content for _, content in found]


def last_boxed(text: str) -> str | None:
    boxes = all_boxed(text)
    return boxes[-1] if boxes else None


def answer_tags(text: str) -> list[str]:
    return [m.strip() for m in _ANSWER_TAG_RE.findall(text)]


def extract_final_answer(text: str, strict: bool = True) -> tuple[str | None, str]:
    """Return ``(answer, method)``; ``answer`` is None if nothing extractable.

    Priority: last <answer> tag (unwrapping a \\boxed inside it) -> last
    \\boxed -> (lenient only) "answer is X" / "#### X" / last number.
    """
    tags = answer_tags(text)
    if tags:
        content = tags[-1]
        boxed = last_boxed(content)
        return (boxed.strip() if boxed is not None else content), "answer_tag"
    boxed = last_boxed(text)
    if boxed is not None:
        return boxed.strip(), "boxed"
    if strict:
        return None, "none"
    m = re.search(r"####\s*(.+)", text)
    if m:
        return m.group(1).strip(), "hash"
    m = re.search(r"answer is[:\s]*\$?([^\n$]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip("."), "answer_is"
    nums = _NUMBER_RE.findall(text)
    if nums:
        return nums[-1], "last_number"
    return None, "none"


def count_distinct_answers(text: str) -> int:
    """Distinct normalized candidates across all tags/boxes (shotgun detection)."""
    cands = [normalize_answer(a) for a in answer_tags(text) + all_boxed(text)]
    return len({c for c in cands if c})


# --------------------------------------------------------------------------
# normalization
# --------------------------------------------------------------------------

_LATEX_REPLACEMENTS = [
    ("\\left", ""),
    ("\\right", ""),
    ("\\!", ""),
    ("\\,", ""),
    ("\\;", ""),
    ("\\dfrac", "\\frac"),
    ("\\tfrac", "\\frac"),
    ("^\\circ", ""),
    ("^{\\circ}", ""),
    ("\\%", ""),
    ("%", ""),
    ("$", ""),
    ("\\$", ""),
]

_UNIT_WORDS = re.compile(
    r"\b(dollars?|cents?|units?|meters?|cm|km|kg|grams?|hours?|minutes?|seconds?|"
    r"days?|years?|apples?|students?|people|items?|miles?|feet|inches?|points?)\b",
    re.IGNORECASE,
)


def _strip_text_cmds(s: str) -> str:
    return re.sub(r"\\(?:text|mathrm|textbf|mbox)\{([^{}]*)\}", r"\1", s)


def normalize_answer(ans: str | None) -> str:
    if ans is None:
        return ""
    s = ans.strip()
    s = _strip_text_cmds(s)
    for old, new in _LATEX_REPLACEMENTS:
        s = s.replace(old, new)
    s = _THOUSANDS_RE.sub(lambda m: m.group(1).replace(",", ""), s)
    s = _UNIT_WORDS.sub("", s)
    s = s.strip().rstrip(".").strip()
    if s.startswith("x=") or s.startswith("x ="):
        s = s.split("=", 1)[1]
    s = re.sub(r"\s+", "", s)
    # \frac12 -> \frac{1}{2}
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


# --------------------------------------------------------------------------
# numeric / symbolic equivalence
# --------------------------------------------------------------------------


def parse_number(s: str) -> Fraction | None:
    """Exact rational parse of ints, decimals, a/b and \\frac{a}{b}."""
    s = normalize_answer(s)
    if not s:
        return None
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m:
        sign = -1 if m.group(1) else 1
        den = int(m.group(3))
        return None if den == 0 else sign * Fraction(int(m.group(2)), den)
    m = re.fullmatch(r"(-?\d+)/(-?\d+)", s)
    if m:
        den = int(m.group(2))
        return None if den == 0 else Fraction(int(m.group(1)), den)
    if re.fullmatch(r"-?(\d+(\.\d*)?|\.\d+)", s):
        return Fraction(s)
    return None


_SAFE_EXPR_RE = re.compile(r"^[0-9a-z+\-*/^().,\s]*$")
_ALLOWED_IDENTS = {"sqrt", "pi", "e", "x", "y", "z", "n", "k", "a", "b", "c", "i", "log", "exp"}
_MAX_EXPR_LEN = 120


def latex_to_sympy_str(s: str) -> str | None:
    """Convert a *small* LaTeX subset to a SymPy-parsable string, or None."""
    s = normalize_answer(s)
    if len(s) > _MAX_EXPR_LEN:
        return None
    for _ in range(4):
        s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
        s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = s.replace("\\pi", "pi").replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("{", "(").replace("}", ")").replace("^", "**")
    if "\\" in s or "__" in s:
        return None
    if not _SAFE_EXPR_RE.match(s):
        return None
    if re.search(r"[a-z)]\s*\.|\.\s*[a-z(]", s):  # attribute access, e.g. "x.y" / "(1).real"
        return None
    idents = set(re.findall(r"[a-z]+", s))
    if not idents <= _ALLOWED_IDENTS:
        return None
    return s


def sympy_equal(a: str, b: str) -> bool:
    """Guarded symbolic equivalence. Never raises; returns False when unsure."""
    sa, sb = latex_to_sympy_str(a), latex_to_sympy_str(b)
    if sa is None or sb is None or not sa or not sb:
        return False
    try:
        import sympy
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )

        tr = standard_transformations + (implicit_multiplication_application,)
        local = {"e": sympy.E, "pi": sympy.pi, "i": sympy.I}
        ea = parse_expr(sa, transformations=tr, local_dict=local, evaluate=True)
        eb = parse_expr(sb, transformations=tr, local_dict=local, evaluate=True)
        if sympy.count_ops(ea) > 60 or sympy.count_ops(eb) > 60:
            return False
        diff = sympy.simplify(ea - eb)
        if diff == 0:
            return True
        if not diff.free_symbols:
            return abs(complex(diff.evalf())) < 1e-9
        return False
    except Exception:  # noqa: BLE001 - any parse/simplify failure means "not equal"
        return False


def _split_top_level(s: str) -> list[str]:
    s = s.strip()
    if len(s) >= 2 and s[0] in "([{" and s[-1] in ")]}":
        s = s[1:-1]
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p for p in parts if p != ""]


def is_equiv(pred: str | None, gold: str | None, tol: float = 1e-6, use_sympy: bool = True) -> bool:
    if pred is None or gold is None:
        return False
    p, g = normalize_answer(pred), normalize_answer(gold)
    if not p or not g:
        return False
    if p == g:
        return True
    # tuples / lists: compare elementwise in order
    if "," in g or "," in p:
        pp, gg = _split_top_level(p), _split_top_level(g)
        if len(pp) == len(gg) > 1:
            return all(is_equiv(x, y, tol, use_sympy) for x, y in zip(pp, gg))
        return False
    pn, gn = parse_number(p), parse_number(g)
    if pn is not None and gn is not None:
        if pn == gn:
            return True
        return abs(float(pn) - float(gn)) <= tol * max(1.0, abs(float(gn)))
    if use_sympy:
        return sympy_equal(p, g)
    return False
