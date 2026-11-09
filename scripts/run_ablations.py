#!/usr/bin/env python
"""Ablation harness: run every variant of one or more ablation files.

Each ablation YAML = a base config (``base:``) + ``variants:`` mapping a name to
dotted overrides. For each variant x seed we train GRPO from the same SFT
checkpoint, evaluate under the shared protocol, and aggregate mean +- std over
seeds into ``experiments/ablations/results/<ablation>.{json,md}``.

    python scripts/run_ablations.py experiments/ablations/remove_kl.yaml --seeds 0 1 2
    python scripts/run_ablations.py experiments/ablations/*.yaml --dry-run
    python scripts/run_ablations.py experiments/ablations/group_size.yaml \
        -o model.name_or_path=tiny-random grpo.steps=5      # CPU smoke
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import statistics
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from reasonrl.config import load_ablation
from reasonrl.pipeline import run_eval, run_grpo

KEYS = ("pass@1", "format_valid_rate", "hack_flag_rate", "mean_completion_chars")


def run_variant(cfg, seed: int, out_root: Path) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg.seed = seed
    tag = cfg.name.replace("/", "__") + f"__s{seed}"
    cfg.grpo.save_dir = str(out_root / "ckpt" / tag)
    ckpt = run_grpo(cfg)
    cfg.eval.checkpoints = {tag: str(ckpt)}
    cfg.eval.output_path = str(out_root / "eval" / f"{tag}.json")
    res = run_eval(cfg)
    row = res["rows"][0]
    return {k: row.get(k) for k in KEYS}


def summarize(per_seed: list[dict]) -> dict:
    out = {}
    for k in KEYS:
        vals = [r[k] for r in per_seed if r.get(k) is not None]
        if vals:
            out[k] = {"mean": statistics.fmean(vals), "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ablations", nargs="+")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--only", nargs="*", default=None, help="subset of variant names")
    ap.add_argument("--out", default="experiments/ablations/results")
    ap.add_argument("--override", "-o", nargs="*", default=[])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    files = [f for pat in args.ablations for f in (glob.glob(pat) or [pat])]
    out_root = Path(args.out)
    for f in files:
        variants = load_ablation(f, args.override)
        name = Path(f).stem
        print(f"== {name}: {[v for v, _ in variants]}")
        results = {}
        for vname, cfg in variants:
            if args.only and vname not in args.only:
                continue
            if args.dry_run:
                g = cfg.grpo
                print(f"  {vname}: kl={g.kl_coef} G={g.group_size} prompts/step={g.prompts_per_step} "
                      f"fmt_w={cfg.reward.format_weight} fmt_mode={cfg.reward.format_mode} "
                      f"guards={cfg.reward.hacking_guards} strict={cfg.reward.strict_extraction}")
                continue
            per_seed = [run_variant(cfg, s, out_root / name) for s in args.seeds]
            results[vname] = {"per_seed": per_seed, "summary": summarize(per_seed)}
        if args.dry_run:
            continue
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / f"{name}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        lines = [f"### {name} (seeds={args.seeds})", "", "| variant | " + " | ".join(KEYS) + " |",
                 "|" + "---|" * (len(KEYS) + 1)]
        for vname, r in results.items():
            cells = [f"{r['summary'][k]['mean']:.3f} ± {r['summary'][k]['std']:.3f}" if k in r["summary"] else "-"
                     for k in KEYS]
            lines.append(f"| {vname} | " + " | ".join(cells) + " |")
        md = "\n".join(lines)
        (out_root / f"{name}.md").write_text(md + "\n", encoding="utf-8")
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
