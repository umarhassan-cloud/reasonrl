import pytest

from reasonrl.config import RewardConfig
from reasonrl.data.dataset import Example
from reasonrl.rewards import RewardFunction, format_score, math_correctness
from reasonrl.utils.math_parse import (
    extract_final_answer,
    is_equiv,
    last_boxed,
    latex_to_sympy_str,
    normalize_answer,
    parse_number,
)


@pytest.mark.parametrize(
    "text,expected,method",
    [
        ("<think>2+2</think><answer>4</answer>", "4", "answer_tag"),
        ("blah \\boxed{\\frac{1}{2}} done", "\\frac{1}{2}", "boxed"),
        ("<answer>The result is \\boxed{12}</answer>", "12", "answer_tag"),
        ("first \\boxed{1} then \\boxed{2}", "2", "boxed"),
        ("\\boxed{\\frac{a}{\\sqrt{b}}}", "\\frac{a}{\\sqrt{b}}", "boxed"),
    ],
)
def test_extraction(text, expected, method):
    assert extract_final_answer(text) == (expected, method)


def test_strict_extraction_refuses_bare_numbers():
    text = "Tom has 3 apples and 5 more so 8"
    assert extract_final_answer(text, strict=True) == (None, "none")
    assert extract_final_answer(text, strict=False) == ("8", "last_number")
    assert extract_final_answer("#### 72", strict=False)[0] == "72"


def test_last_boxed_nested_braces():
    assert last_boxed("x \\boxed{\\frac{1}{2}+{3}} y") == "\\frac{1}{2}+{3}"
    assert last_boxed("unterminated \\boxed{12") is None


@pytest.mark.parametrize(
    "pred,gold",
    [
        ("8", "8"),
        ("8.0", "8"),
        ("1,000", "1000"),
        ("$5", "5"),
        ("\\frac{1}{2}", "0.5"),
        ("1/2", "\\frac{1}{2}"),
        ("\\dfrac{3}{4}", "\\frac{3}{4}"),
        ("12 apples", "12"),
        ("x = 3", "3"),
        ("\\text{7}", "7"),
        ("2\\sqrt{2}", "\\sqrt{8}"),
        ("(1, 2)", "(1,2)"),
        ("50\\%", "50"),
    ],
)
def test_equivalent(pred, gold):
    assert is_equiv(pred, gold)


@pytest.mark.parametrize("pred,gold", [("7", "8"), ("1/3", "0.33"), ("", "0"), (None, "1"), ("(1,2)", "(2,1)")])
def test_not_equivalent(pred, gold):
    assert not is_equiv(pred, gold)


def test_parse_number_exact_rationals():
    assert parse_number("\\frac{6}{8}") == parse_number("3/4") == parse_number("0.75")
    assert parse_number("1/0") is None


def test_sympy_guard_rejects_code_injection():
    # SymPy's parser evals strings; anything outside the whitelist must never reach it.
    for evil in ["__import__('os').system('echo hi')", "lambda: 1", "open('x')", "a.b", "x;y"]:
        assert latex_to_sympy_str(evil) is None
    assert not is_equiv("__import__('os')", "0")


def test_normalize_answer():
    assert normalize_answer(" \\left( 3 \\right). ") == "(3)"
    assert normalize_answer("\\frac12") == "\\frac{1}{2}"


def test_math_correctness_end_to_end():
    score, pred, info = math_correctness("<think>3+5=8</think>\n<answer>8</answer>", "8")
    assert score == 1.0 and pred == "8" and info["extract_method"] == "answer_tag"
    assert math_correctness("<answer>9</answer>", "8")[0] == 0.0


def test_format_strict_vs_naive():
    good = "<think>3 plus 5 is 8 here</think>\n<answer>8</answer>"
    empty = "<think></think><answer></answer>"
    assert format_score(good, "math", "strict")[0] == 1.0
    assert format_score(empty, "math", "strict")[0] == 0.0
    assert format_score(empty, "math", "naive")[0] == 1.0  # the hackable variant
    assert format_score(good + " trailing junk", "math", "strict")[0] == 0.5
    assert format_score("<answer>1</answer><answer>2</answer>", "math", "strict")[0] == 0.0


def test_composite_reward_weights_and_gating():
    ex = Example(id="m1", task="math", question="What is 3 + 5?", answer="8")
    rf = RewardFunction(RewardConfig(correctness_weight=1.0, format_weight=0.1))
    out = rf(ex, "<think>3 plus 5 is 8 here</think>\n<answer>8</answer>")
    assert out.total == pytest.approx(1.1) and out.correctness == 1.0 and not out.guarded
    # correct but malformed (no think) -> partial format
    out = rf(ex, "<answer>8</answer>")
    assert out.total == pytest.approx(1.05)
    gated = RewardFunction(RewardConfig(format_weight=0.1, require_format_for_correctness=True))
    assert gated(ex, "<answer>8</answer>").total == pytest.approx(0.05)
