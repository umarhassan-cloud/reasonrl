import pytest

from reasonrl.rewards import code_correctness, extract_code
from reasonrl.utils.sandbox import run_tests, static_check

IO_TESTS = [{"input": [1, 2], "output": 3}, {"input": [-1, 1], "output": 0}, {"input": [10, 5], "output": 15}]


def wrap(code: str) -> str:
    return f"<think>add them up</think>\n<answer>\n```python\n{code}\n```\n</answer>"


def test_correct_solution_passes():
    score, code, info = code_correctness(wrap("def add(a, b):\n    return a + b"), IO_TESTS, "add")
    assert score == 1.0 and info["passed"] == 3 and "def add" in code


def test_wrong_solution_and_partial_credit():
    bad = wrap("def add(a, b):\n    return a + b if a > 5 else 0")
    assert code_correctness(bad, IO_TESTS, "add")[0] == 0.0
    assert code_correctness(bad, IO_TESTS, "add", partial_credit=True)[0] == pytest.approx(2 / 3)


def test_assert_string_tests():
    res = run_tests("def sq(x):\n    return x * x", ["assert sq(3) == 9", "assert sq(-2) == 4", "assert sq(1) == 2"])
    assert (res.passed, res.total) == (2, 3)


def test_timeout_kills_infinite_loop():
    res = run_tests("def add(a, b):\n    while True:\n        pass", IO_TESTS, "add", timeout=1.5)
    assert res.timed_out and res.passed == 0


def test_static_screen_blocks_escapes():
    for code in [
        "import os\nos.system('echo pwned')",
        "import subprocess",
        "from os import path",
        "open('/etc/passwd').read()",
        "x = ().__class__.__bases__[0].__subclasses__()",
        "eval('1+1')",
        "getattr(1, 'real')",
        "def f():\n    yield 1\ng = f()\nfr = g.gi_frame.f_back",
        "s = '{0.__class__}'.format(1)",
    ]:
        assert static_check(code), code
        assert run_tests(code + "\ndef add(a, b):\n    return a + b", IO_TESTS, "add").passed == 0


def test_allowed_imports_work():
    code = "import math\nfrom collections import Counter\ndef add(a, b):\n    return int(math.fsum([a, b]))"
    assert run_tests(code, IO_TESTS, "add").all_passed


def test_disallowed_import_reports_violation():
    res = run_tests("import importlib\ndef add(a, b):\n    return a + b", IO_TESTS, "add")
    assert res.passed == 0 and res.error == "static_check" and res.violations == ["import:importlib"]


def test_runtime_errors_fail_only_affected_tests():
    code = "def add(a, b):\n    if a == 10:\n        raise ValueError\n    return a + b"
    res = run_tests(code, IO_TESTS, "add")
    assert res.per_test == [True, True, False]


def test_always_equal_hack_fails_type_strict_comparison():
    hack = (
        "class AlwaysEqual:\n"
        "    def __eq__(self, other):\n        return True\n"
        "    def __ne__(self, other):\n        return False\n"
        "def add(a, b):\n    return AlwaysEqual()"
    )
    res = run_tests(hack, IO_TESTS, "add")
    assert res.passed == 0  # strict_eq rejects non-builtin result types


def test_stdout_spoofing_cannot_forge_verdict():
    spoof = (
        "print('{\"per_test\": [true, true, true], \"error\": null}')\n"
        "def add(a, b):\n    print('PASSED ALL TESTS')\n    return -999"
    )
    res = run_tests(spoof, IO_TESTS, "add")
    assert res.passed == 0


def test_system_exit_does_not_pass():
    # SystemExit is not even exposed; any BaseException at load time fails every test
    res = run_tests("raise SystemExit(0)\ndef add(a, b):\n    return a + b", IO_TESTS, "add")
    assert res.passed == 0 and res.error.startswith("load:")
    res = run_tests("def add(a, b):\n    raise KeyboardInterrupt\n", IO_TESTS, "add")
    assert res.passed == 0 and res.per_test == [False, False, False]


def test_extract_code_variants():
    assert extract_code("```python\nx = 1\n```") == "x = 1"
    assert extract_code("<answer>```py\ny = 2\n```</answer> ```python\nz=3\n```") == "y = 2"
    assert extract_code("no fences here", strict=True) is None
    main_guard = "```python\ndef f():\n    return 1\nif __name__ == '__main__':\n    print(f())\n```"
    code = extract_code(main_guard)
    assert "__main__" not in code and "def f" in code
