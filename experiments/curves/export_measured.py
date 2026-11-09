#!/usr/bin/env python
"""Copy a run's raw ``metrics.jsonl`` here (unchanged) and add a CSV view.

    python experiments/curves/export_measured.py runs/smoke/smoke/grpo/metrics.jsonl cpu_smoke_grpo
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent


def main(src: str, name: str) -> None:
    rows = [json.loads(line) for line in Path(src).read_text(encoding="utf-8").splitlines() if line.strip()]
    shutil.copyfile(src, OUT / f"{name}.jsonl")
    keys: list[str] = []
    for r in rows:
        keys.extend(k for k in r if k not in keys)
    with open(OUT / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{name}: {len(rows)} records, {len(keys)} columns")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
