#!/usr/bin/env python
"""Stage 1: supervised fine-tuning on curated reasoning traces.

    python scripts/train_sft.py --config configs/sft.yaml
    python scripts/train_sft.py --config configs/smoke/sft_tiny.yaml   # CPU, ~1 min
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
from reasonrl.config import load_config
from reasonrl.pipeline import run_sft


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", "-o", nargs="*", default=[], help="dotted overrides, e.g. optim.lr=2e-5")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.override)
    print(f"saved SFT checkpoint -> {run_sft(cfg)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
