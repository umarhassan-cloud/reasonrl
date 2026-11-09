#!/usr/bin/env python
"""Stage 2b: DPO on verifier-labelled preference pairs sampled from the SFT model.

    python scripts/train_dpo.py --config configs/dpo.yaml
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
from reasonrl.config import load_config
from reasonrl.pipeline import run_dpo


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", "-o", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.override)
    print(f"saved DPO checkpoint -> {run_dpo(cfg)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
