#!/usr/bin/env python
"""Build datasets in the reasonrl JSONL schema.

Subcommands
-----------
synthetic   Regenerate ``data/sample/*`` deterministically (CPU demo data).
            Train and eval math problems use *disjoint templates*; a few eval
            items are deliberately planted copies of train items so the
            contamination checker has something real to find.
hf          Convert GSM8K / MATH / MBPP from the HuggingFace Hub (needs
            ``datasets`` and network) into the same schema.

Schema: {"id", "task", "question", "answer"?, "solution"?, "tests"?,
         "entry_point"?, "source", "meta"}
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from fractions import Fraction
from pathlib import Path

import _bootstrap  # noqa: F401
from reasonrl.data.dataset import write_jsonl

NAMES = ["Ava", "Ben", "Chen", "Dara", "Eli", "Fatima", "Goran", "Hana", "Ivan", "Jia"]


def _sol(think: str, answer: str) -> str:
    return f"<think>{think}</think>\n<answer>{answer}</answer>"


# ---------------------------------------------------------------- math: train
def t_add(r):
    n, a, b = r.choice(NAMES), r.randint(2, 40), r.randint(2, 40)
    q = f"{n} has {a} apples and buys {b} more. How many apples does {n} have now?"
    return q, str(a + b), f"{n} starts with {a} apples and buys {b} more, so {a} + {b} = {a + b}."


def t_boxes(r):
    a, b = r.randint(3, 15), r.randint(2, 9)
    q = f"A box holds {a} pencils. How many pencils are in {b} boxes?"
    return q, str(a * b), f"Each box has {a} pencils and there are {b} boxes, so {a} * {b} = {a * b}."


def t_spend(r):
    n, b = r.choice(NAMES), r.randint(2, 30)
    a = b + r.randint(1, 50)
    q = f"{n} had {a} dollars and spent {b} dollars on lunch. How many dollars are left?"
    return q, str(a - b), f"{n} spends {b} of {a} dollars, so {a} - {b} = {a - b}."


def t_share(r):
    b, k = r.randint(2, 9), r.randint(2, 12)
    a = b * k
    q = f"{a} cookies are shared equally among {b} friends. How many cookies does each friend get?"
    return q, str(k), f"Divide {a} cookies by {b} friends: {a} / {b} = {k}."


def t_order(r):
    a, b, c = r.randint(1, 20), r.randint(2, 9), r.randint(2, 9)
    q = f"What is {a} + {b} * {c}?"
    return q, str(a + b * c), f"Multiplication first: {b} * {c} = {b * c}, then {a} + {b * c} = {a + b * c}."


def t_area(r):
    a, b = r.randint(2, 20), r.randint(2, 20)
    q = f"A rectangle has length {a} and width {b}. What is its area?"
    return q, str(a * b), f"Area is length times width: {a} * {b} = {a * b}."


def t_percent(r):
    a, b = r.choice([10, 20, 25, 50]), 20 * r.randint(1, 20)
    q = f"What is {a} percent of {b}?"
    v = a * b // 100
    return q, str(v), f"{a} percent of {b} is {a}/100 * {b} = {v}."


def t_linear(r):
    a, b = r.randint(1, 30), r.randint(31, 90)
    q = f"Solve for x: x + {a} = {b}."
    return q, str(b - a), f"Subtract {a} from both sides: x = {b} - {a} = {b - a}."


def t_speed(r):
    a, b = 10 * r.randint(3, 12), r.randint(2, 6)
    q = f"A train travels {a} km per hour for {b} hours. How far does it travel in km?"
    return q, str(a * b), f"Distance is speed times time: {a} * {b} = {a * b}."


def t_triangular(r):
    n = r.randint(3, 30)
    q = f"What is the sum of the first {n} positive integers?"
    v = n * (n + 1) // 2
    return q, str(v), f"Use n(n+1)/2 with n = {n}: {n} * {n + 1} / 2 = {v}."


# ---------------------------------------------------------------- math: eval (disjoint templates)
def e_eggs(r):
    a, b = r.randint(4, 24), r.randint(3, 9)
    q = f"A farmer collects {a} eggs each day. How many eggs are collected in {b} days?"
    return q, str(a * b), f"{a} eggs per day for {b} days gives {a} * {b} = {a * b}."


def e_pages(r):
    a, b, c = r.randint(5, 40), r.randint(5, 40), r.randint(5, 40)
    q = (f"Maria reads {a} pages on Monday, {b} pages on Tuesday and {c} pages on Wednesday. "
         "How many pages did she read in total?")
    return q, str(a + b + c), f"Add the three days: {a} + {b} + {c} = {a + b + c}."


def e_scale(r):
    a, k = r.randint(2, 12), r.randint(2, 15)
    q = f"Solve for x: {a}x = {a * k}."
    return q, str(k), f"Divide both sides by {a}: x = {a * k} / {a} = {k}."


def e_mod(r):
    b = r.randint(3, 12)
    a = r.randint(20, 200)
    q = f"What is the remainder when {a} is divided by {b}?"
    return q, str(a % b), f"{a} = {b} * {a // b} + {a % b}, so the remainder is {a % b}."


def e_sale(r):
    a, p = 20 * r.randint(1, 10), r.choice([10, 25, 50])
    v = a * (100 - p) // 100
    q = f"A shirt costs {a} dollars and is on sale for {p} percent off. What is the sale price in dollars?"
    return q, str(v), f"The discount is {p}/100 * {a} = {a * p // 100}, so the price is {a} - {a * p // 100} = {v}."


def e_frac(r):
    a, b, c, d = r.randint(1, 5), r.randint(2, 9), r.randint(1, 5), r.randint(2, 9)
    v = Fraction(a, b) + Fraction(c, d)
    ans = str(v.numerator) if v.denominator == 1 else f"\\frac{{{v.numerator}}}{{{v.denominator}}}"
    q = f"What is {a}/{b} + {c}/{d}? Give the answer as a simplified fraction."
    return q, ans, f"Common denominator {b * d}: {a * d}/{b * d} + {c * b}/{b * d} = {v}."


TRAIN_TEMPLATES = [t_add, t_boxes, t_spend, t_share, t_order, t_area, t_percent, t_linear, t_speed, t_triangular]
EVAL_TEMPLATES = [e_eggs, e_pages, e_scale, e_mod, e_sale, e_frac]


def make_math(seed: int, per_train: int = 6, per_eval: int = 4):
    r = random.Random(seed)
    train, ev = [], []
    for t in TRAIN_TEMPLATES:
        for i in range(per_train):
            q, a, th = t(r)
            train.append({"id": f"math-train-{t.__name__[2:]}-{i}", "task": "math", "question": q,
                          "answer": a, "solution": _sol(th, a), "source": "synthetic", "meta": {"template": t.__name__}})
    for t in EVAL_TEMPLATES:
        for i in range(per_eval):
            q, a, th = t(r)
            ev.append({"id": f"math-eval-{t.__name__[2:]}-{i}", "task": "math", "question": q,
                       "answer": a, "solution": _sol(th, a), "source": "synthetic", "meta": {"template": t.__name__}})
    # Planted contamination: verbatim copies of train items (caught by plain n-grams) ...
    for k, src in enumerate([train[3], train[25], train[47]]):
        ev.append({**src, "id": f"math-eval-x{k}", "meta": {"template": src["meta"]["template"], "planted": "verbatim"}})
    # ... and a templated near-duplicate with new numbers (caught only with mask_numbers).
    q, a, th = t_speed(random.Random(seed + 1))
    ev.append({"id": "math-eval-x3", "task": "math", "question": q, "answer": a, "solution": _sol(th, a),
               "source": "synthetic", "meta": {"template": "t_speed", "planted": "template"}})
    return train, ev


def make_toy_addition(seed: int, n_eval: int = 30):
    """Single-digit addition, split by *fact*: eval pairs (a, b) never appear in train.

    In-distribution toy where a tiny CPU model can actually move held-out
    accuracy, so GRPO/DPO effects are measurable without a GPU.
    """
    r = random.Random(seed)
    pairs = [(a, b) for a in range(10) for b in range(10)]
    r.shuffle(pairs)
    rows = []
    for a, b in pairs:
        rows.append({"id": f"add-{a}-{b}", "task": "math", "question": f"What is {a} + {b}?", "answer": str(a + b),
                     "solution": _sol(f"{a} + {b} = {a + b}", str(a + b)), "source": "toy_addition"})
    return rows[n_eval:], rows[:n_eval]


# ---------------------------------------------------------------- code tasks
CODE_TASKS = [
    # (entry_point, description, solution, tests, split)
    ("add_numbers", "Write a function add_numbers(a, b) that returns the sum of two integers.",
     "def add_numbers(a, b):\n    return a + b",
     [[[1, 2], 3], [[-5, 5], 0], [[100, 23], 123]], "train"),
    ("reverse_string", "Write a function reverse_string(s) that returns the string s reversed.",
     "def reverse_string(s):\n    return s[::-1]",
     [[["abc"], "cba"], [[""], ""], [["racecar"], "racecar"], [["ab cd"], "dc ba"]], "train"),
    ("is_palindrome", "Write a function is_palindrome(s) that returns True if the lowercase string s reads the "
     "same forwards and backwards, else False.",
     "def is_palindrome(s):\n    return s == s[::-1]",
     [[["level"], True], [["hello"], False], [[""], True], [["ab"], False]], "train"),
    ("factorial", "Write a function factorial(n) that returns n! for a non-negative integer n.",
     "def factorial(n):\n    out = 1\n    for i in range(2, n + 1):\n        out *= i\n    return out",
     [[[0], 1], [[1], 1], [[5], 120], [[10], 3628800]], "train"),
    ("max_of_list", "Write a function max_of_list(xs) that returns the largest element of a non-empty list of "
     "integers without using the built-in max.",
     "def max_of_list(xs):\n    best = xs[0]\n    for x in xs[1:]:\n        if x > best:\n            best = x\n"
     "    return best",
     [[[[3, 1, 2]], 3], [[[-4, -2, -9]], -2], [[[7]], 7]], "train"),
    ("count_vowels", "Write a function count_vowels(s) that returns how many vowels (a, e, i, o, u) appear in "
     "the lowercase string s.",
     "def count_vowels(s):\n    return sum(1 for c in s if c in 'aeiou')",
     [[["hello"], 2], [["rhythm"], 0], [["aeiou"], 5], [[""], 0]], "train"),
    ("sum_of_evens", "Write a function sum_of_evens(xs) that returns the sum of the even integers in the list xs.",
     "def sum_of_evens(xs):\n    return sum(x for x in xs if x % 2 == 0)",
     [[[[1, 2, 3, 4]], 6], [[[]], 0], [[[1, 3, 5]], 0], [[[-2, 2, 7]], 0]], "train"),
    ("fibonacci", "Write a function fibonacci(n) that returns the n-th Fibonacci number with fibonacci(0) = 0 "
     "and fibonacci(1) = 1.",
     "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
     [[[0], 0], [[1], 1], [[2], 1], [[10], 55], [[20], 6765]], "train"),
    ("celsius_to_fahrenheit", "Write a function celsius_to_fahrenheit(c) that converts a temperature from "
     "Celsius to Fahrenheit.",
     "def celsius_to_fahrenheit(c):\n    return c * 9 / 5 + 32",
     [[[0], 32.0], [[100], 212.0], [[-40], -40.0]], "train"),
    ("unique_sorted", "Write a function unique_sorted(xs) that returns the distinct elements of xs in "
     "ascending order as a list.",
     "def unique_sorted(xs):\n    return sorted(set(xs))",
     [[[[3, 1, 3, 2]], [1, 2, 3]], [[[]], []], [[[5, 5, 5]], [5]]], "train"),
    ("gcd", "Implement gcd(a, b) returning the greatest common divisor of two positive integers.",
     "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
     [[[12, 18], 6], [[7, 13], 1], [[100, 75], 25], [[9, 9], 9]], "eval"),
    ("is_prime", "Implement is_prime(n) that returns True when the integer n is a prime number.",
     "def is_prime(n):\n    if n < 2:\n        return False\n    i = 2\n    while i * i <= n:\n"
     "        if n % i == 0:\n            return False\n        i += 1\n    return True",
     [[[2], True], [[1], False], [[17], True], [[21], False], [[97], True]], "eval"),
    ("second_largest", "Implement second_largest(xs) that returns the second largest distinct value in a list "
     "with at least two distinct integers.",
     "def second_largest(xs):\n    vals = sorted(set(xs))\n    return vals[-2]",
     [[[[1, 5, 3]], 3], [[[4, 4, 2]], 2], [[[-1, -7, -3]], -3]], "eval"),
    ("word_count", "Implement word_count(s) that returns the number of whitespace-separated words in s.",
     "def word_count(s):\n    return len(s.split())",
     [[["hello world"], 2], [[""], 0], [["  a  b c "], 3]], "eval"),
    ("flatten", "Implement flatten(xss) that flattens a list of lists by one level.",
     "def flatten(xss):\n    return [x for xs in xss for x in xs]",
     [[[[[1, 2], [3]]], [1, 2, 3]], [[[]], []], [[[[], [4]]], [4]]], "eval"),
    ("run_length_encode", "Implement run_length_encode(s) that compresses runs of characters, e.g. 'aaabcc' "
     "becomes 'a3b1c2'.",
     "def run_length_encode(s):\n    out = []\n    i = 0\n    while i < len(s):\n        j = i\n"
     "        while j < len(s) and s[j] == s[i]:\n            j += 1\n        out.append(s[i] + str(j - i))\n"
     "        i = j\n    return ''.join(out)",
     [[["aaabcc"], "a3b1c2"], [[""], ""], [["z"], "z1"]], "eval"),
]


def make_code():
    train, ev = [], []
    for ep, desc, sol, tests, split in CODE_TASKS:
        row = {
            "id": f"code-{split}-{ep}",
            "task": "code",
            "question": desc,
            "entry_point": ep,
            "tests": [{"input": i, "output": o} for i, o in tests],
            "solution": "<think>Implement the function directly.</think>\n<answer>\n```python\n" + sol + "\n```\n</answer>",
            "source": "handwritten",
            "meta": {},
        }
        (train if split == "train" else ev).append(row)
    leak = next(r for r in train if r["entry_point"] == "count_vowels")
    ev.append({**leak, "id": "code-eval-x0", "meta": {"planted": "verbatim"}})
    return train, ev


# ---------------------------------------------------------------- HF conversion
def _gsm8k_answer(ans: str) -> str:
    return ans.split("####")[-1].strip().replace(",", "")


def convert_hf(name: str, out_dir: Path, limit: int | None) -> None:
    from datasets import load_dataset  # optional dependency

    from reasonrl.utils.math_parse import last_boxed

    if name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main")
        for split, out in (("train", "gsm8k_train"), ("test", "gsm8k_eval")):
            rows = []
            for i, ex in enumerate(ds[split]):
                if limit and i >= limit:
                    break
                ans = _gsm8k_answer(ex["answer"])
                think = re.sub(r"<<.*?>>", "", ex["answer"].split("####")[0]).strip()
                rows.append({"id": f"gsm8k-{split}-{i}", "task": "math", "question": ex["question"],
                             "answer": ans, "solution": _sol(think, ans), "source": "gsm8k"})
            write_jsonl(out_dir / f"{out}.jsonl", rows)
    elif name == "math":
        ds = load_dataset("EleutherAI/hendrycks_math", "algebra")
        for split, out in (("train", "math_algebra_train"), ("test", "math_algebra_eval")):
            rows = []
            for i, ex in enumerate(ds[split]):
                if limit and i >= limit:
                    break
                ans = last_boxed(ex["solution"]) or ""
                rows.append({"id": f"math-{split}-{i}", "task": "math", "question": ex["problem"], "answer": ans,
                             "solution": _sol(ex["solution"], ans), "source": "hendrycks_math",
                             "meta": {"level": ex.get("level")}})
            write_jsonl(out_dir / f"{out}.jsonl", rows)
    elif name == "mbpp":
        ds = load_dataset("google-research-datasets/mbpp", "sanitized")
        for split, out in (("train", "mbpp_train"), ("test", "mbpp_eval")):
            rows = []
            for i, ex in enumerate(ds[split]):
                if limit and i >= limit:
                    break
                m = re.search(r"def\s+(\w+)\s*\(", ex["code"])
                rows.append({"id": f"mbpp-{split}-{ex['task_id']}", "task": "code", "question": ex["prompt"],
                             "entry_point": m.group(1) if m else None, "tests": ex["test_list"],
                             "solution": "<think>Implement the function.</think>\n<answer>\n```python\n"
                             + ex["code"].strip() + "\n```\n</answer>", "source": "mbpp"})
            write_jsonl(out_dir / f"{out}.jsonl", rows)
    else:
        raise SystemExit(f"unknown dataset {name}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synthetic")
    s.add_argument("--out", default="data/sample")
    s.add_argument("--seed", type=int, default=1234)
    h = sub.add_parser("hf")
    h.add_argument("--dataset", choices=["gsm8k", "math", "mbpp"], required=True)
    h.add_argument("--out", default="data/hf")
    h.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    if args.cmd == "synthetic":
        out = Path(args.out)
        mt, me = make_math(args.seed)
        ct, ce = make_code()
        write_jsonl(out / "math_train.jsonl", mt)
        write_jsonl(out / "math_eval.jsonl", me)
        write_jsonl(out / "code_train.jsonl", ct)
        write_jsonl(out / "code_eval.jsonl", ce)
        at, ae = make_toy_addition(args.seed)
        write_jsonl(out / "toy_add_train.jsonl", at)
        write_jsonl(out / "toy_add_eval.jsonl", ae)
        print(f"math: {len(mt)} train / {len(me)} eval; code: {len(ct)} train / {len(ce)} eval; "
              f"toy addition: {len(at)} / {len(ae)} -> {out}")
    else:
        convert_hf(args.dataset, Path(args.out), args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
