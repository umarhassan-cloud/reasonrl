#!/usr/bin/env python
"""Evaluate checkpoints under one shared protocol on the clean held-out set.

    python scripts/evaluate.py --config configs/eval.yaml
    python scripts/evaluate.py --config configs/eval.yaml -o eval.n_samples=8 eval.temperature=0.7 "eval.k_values=[1,8]"
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
from reasonrl.config import load_config
from reasonrl.pipeline import run_eval


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", "-o", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.override)
    out = run_eval(cfg)
    print(out["table_markdown"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
