"""Verifiable code reward: run hidden unit tests in the sandbox."""

from __future__ import annotations

import ast
import re

from reasonrl.utils.sandbox import SandboxResult, run_tests

_FENCE_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _drop_main_guard(code: str) -> str:
    """Remove top-level ``if __name__ == "__main__":`` blocks (demo code, not solution)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    body = []
    for node in tree.body:
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
        ):
            continue
        body.append(node)
    if len(body) == len(tree.body):
        return code
    tree.body = body
    return ast.unparse(tree)


def extract_code(completion: str, strict: bool = True) -> str | None:
    """Last fenced code block, preferring one inside ``<answer>``."""
    scope = completion
    answers = _ANSWER_RE.findall(completion)
    if answers:
        scope = answers[-1]
    blocks = _FENCE_RE.findall(scope) or _FENCE_RE.findall(completion)
    if blocks:
        return _drop_main_guard(blocks[-1].strip())
    if strict:
        return None
    return _drop_main_guard(scope.strip()) or None


def code_correctness(
    completion: str,
    tests: list,
    entry_point: str | None,
    timeout: float = 3.0,
    partial_credit: bool = False,
    strict: bool = True,
) -> tuple[float, str | None, dict]:
    """Return ``(score, extracted_code, info)``.

    Binary all-tests-pass by default: partial credit rewards solutions that
    special-case the easy tests, which we found to be a weak hacking vector.
    """
    code = extract_code(completion, strict=strict)
    if code is None:
        return 0.0, None, {"error": "no_code"}
    res: SandboxResult = run_tests(code, tests, entry_point=entry_point, timeout=timeout)
    score = res.pass_rate if partial_credit else float(res.all_passed)
    info = {
        "passed": res.passed,
        "total": res.total,
        "error": res.error,
        "timed_out": res.timed_out,
        "violations": res.violations,
    }
    return score, code, info
