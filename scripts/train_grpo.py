#!/usr/bin/env python
"""Stage 2a: GRPO with verifiable rewards, initialized from the SFT checkpoint.

    python scripts/train_grpo.py --config configs/grpo.yaml
    python scripts/train_grpo.py --config configs/grpo.yaml -o grpo.kl_coef=0.0 name=grpo_nokl
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
from reasonrl.config import load_config
from reasonrl.pipeline import run_grpo


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", "-o", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.override)
    print(f"saved GRPO checkpoint -> {run_grpo(cfg)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
