#!/usr/bin/env python
"""Generate the ILLUSTRATIVE training curves in this directory.

These curves are *synthesized* from simple parametric shapes + seeded noise.
They are NOT measurements. They encode the qualitative behaviour we expect at
GPU scale (0.5B-1.5B model, GSM8K/MBPP-style data, configs/grpo.yaml) and that
is reported in the GRPO / DAPO / Dr. GRPO literature, so that plotting code,
dashboards and the write-up can be developed before a GPU run exists.
Every file carries ``"provenance": "illustrative"``. Measured CPU smoke-run
curves live next to them with ``"provenance": "measured"``.

    python experiments/curves/make_illustrative_curves.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
STEPS = np.arange(0, 201, 5)
META = {
    "provenance": "illustrative",
    "note": "Synthesized shapes, not measured. See make_illustrative_curves.py and experiments/RESULTS.md.",
    "assumed_setup": "Qwen2.5-0.5B-Instruct-class policy, SFT init, 8 prompts x G=8, lr 1e-6, 200 steps",
}


def sat(t, lo, hi, tau):
    return lo + (hi - lo) * (1 - np.exp(-t / tau))


def noisy(x, scale, rng):
    return np.round(x + rng.normal(0, scale, size=x.shape), 4)


def grpo_runs(rng):
    t = STEPS.astype(float)
    runs = {
        "grpo_kl0.04": {
            "reward/correct": noisy(sat(t, 0.34, 0.55, 70), 0.015, rng),
            "eval/pass@1": noisy(sat(t, 0.31, 0.44, 80), 0.01, rng),
            "train/kl": noisy(sat(t, 0.0, 0.07, 90), 0.003, rng).clip(0),
            "completion/len_mean": noisy(sat(t, 182, 236, 120), 4, rng),
            "reward/format_valid_rate": noisy(sat(t, 0.86, 0.98, 40), 0.008, rng).clip(0, 1),
            "reward/guarded_frac": noisy(0.01 + 0 * t, 0.004, rng).clip(0),
        },
        "grpo_nokl": {
            "reward/correct": noisy(sat(t, 0.34, 0.60, 45), 0.018, rng),
            "eval/pass@1": noisy(sat(t, 0.31, 0.42, 50) - 0.00012 * np.maximum(t - 100, 0), 0.012, rng),
            "train/kl": noisy(0.0035 * t ** 1.05, 0.01, rng).clip(0),  # logged vs ref, not optimized
            "completion/len_mean": noisy(182 + 1.15 * t, 6, rng),
            "reward/format_valid_rate": noisy(sat(t, 0.86, 0.97, 30) - 0.0003 * np.maximum(t - 120, 0), 0.01, rng),
            "reward/guarded_frac": noisy(0.01 + 0.0004 * t, 0.006, rng).clip(0),
        },
        "grpo_nokl_unguarded": {
            "reward/correct": noisy(sat(t, 0.34, 0.66, 40), 0.02, rng),
            "eval/pass@1": noisy(sat(t, 0.31, 0.39, 40) - 0.0003 * np.maximum(t - 80, 0), 0.012, rng),
            "train/kl": noisy(0.004 * t ** 1.08, 0.012, rng).clip(0),
            "completion/len_mean": noisy(182 + 1.4 * t, 7, rng),
            "reward/format_valid_rate": noisy(sat(t, 0.86, 0.96, 30), 0.01, rng),
            "hack/any_flag_rate": noisy(0.01 + 0.11 / (1 + np.exp(-(t - 110) / 15)), 0.006, rng).clip(0),
        },
        "grpo_g4_budget64": {
            "reward/correct": noisy(sat(t, 0.34, 0.53, 75), 0.02, rng),
            "eval/pass@1": noisy(sat(t, 0.31, 0.425, 85), 0.012, rng),
            "groups/zero_std_frac": noisy(sat(t, 0.46, 0.55, 100), 0.02, rng).clip(0, 1),
        },
        "grpo_g8_budget64": {
            "reward/correct": noisy(sat(t, 0.34, 0.55, 70), 0.015, rng),
            "eval/pass@1": noisy(sat(t, 0.31, 0.44, 80), 0.01, rng),
            "groups/zero_std_frac": noisy(sat(t, 0.30, 0.38, 100), 0.015, rng).clip(0, 1),
        },
    }
    return {k: {m: v.tolist() for m, v in d.items()} for k, d in runs.items()}


def dpo_run(rng):
    steps = np.arange(0, 61, 2).astype(float)
    return steps, {
        "train/loss": noisy(math.log(2) * np.exp(-steps / 45) + 0.25 * (1 - np.exp(-steps / 45)), 0.015, rng),
        "train/rewards/accuracy": noisy(sat(steps, 0.5, 0.86, 18), 0.02, rng).clip(0, 1),
        "train/rewards/margin": noisy(sat(steps, 0.0, 1.9, 25), 0.05, rng),
        "train/logps/chosen": noisy(-62 - 0.05 * steps, 0.5, rng),
        "train/logps/rejected": noisy(-64 - 0.22 * steps, 0.6, rng),
    }


def write(name: str, steps, series: dict) -> None:
    with open(OUT / f"{name}.json", "w", encoding="utf-8") as fh:
        json.dump({**META, "run": name, "steps": [int(s) for s in steps], "series": series}, fh, indent=1)
    with open(OUT / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        keys = list(series)
        w.writerow(["step", *keys])
        for i, s in enumerate(steps):
            w.writerow([int(s), *[series[k][i] for k in keys]])


def main() -> None:
    rng = np.random.default_rng(20260101)
    runs = grpo_runs(rng)
    with open(OUT / "illustrative_grpo_runs.json", "w", encoding="utf-8") as fh:
        json.dump({**META, "steps": STEPS.tolist(), "runs": runs}, fh, indent=1)
    write("illustrative_grpo_kl0.04", STEPS, runs["grpo_kl0.04"])
    write("illustrative_grpo_nokl", STEPS, runs["grpo_nokl"])
    steps, dpo = dpo_run(rng)
    write("illustrative_dpo", steps, {k: v.tolist() for k, v in dpo.items()})
    print(f"wrote illustrative curves to {OUT}")


if __name__ == "__main__":
    main()
