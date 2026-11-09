#!/usr/bin/env python
"""Report train/eval n-gram contamination and write a clean eval file.

    python scripts/check_contamination.py \
        --train data/sample/math_train.jsonl data/sample/code_train.jsonl \
        --eval data/sample/math_eval.jsonl data/sample/code_eval.jsonl \
        --ngram 8 --threshold 0.5 --mask-numbers --write-clean data/sample/clean_eval.jsonl

Exit code 1 with ``--fail-above RATE`` lets CI block contaminated eval sets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from reasonrl.data.dataset import load_examples, write_jsonl
from reasonrl.eval.contamination import contamination_for_examples, split_clean


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--ngram", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--mask-numbers", action="store_true", help="also catch templated near-duplicates")
    ap.add_argument("--write-clean", default=None)
    ap.add_argument("--report", default=None, help="write JSON report here")
    ap.add_argument("--fail-above", type=float, default=None)
    args = ap.parse_args(argv)

    train = load_examples(args.train)
    ev = load_examples(args.eval)
    report = contamination_for_examples(train, ev, args.ngram, args.threshold, args.mask_numbers)
    print(report.to_markdown())
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    if args.write_clean:
        clean, _ = split_clean(ev, report)
        write_jsonl(args.write_clean, clean)
        print(f"\nwrote {len(clean)} clean eval items -> {args.write_clean}")
    if args.fail_above is not None and report.rate > args.fail_above:
        print(f"contamination rate {report.rate:.3f} > {args.fail_above}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
