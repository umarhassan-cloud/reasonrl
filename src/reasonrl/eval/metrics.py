"""Statistics for honest comparisons on small eval sets."""

from __future__ import annotations

import math

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021): 1 - C(n-c, k) / C(n, k)."""
    if k > n:
        raise ValueError(f"k={k} > n={n}")
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_ci(values: list[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """(mean, lo, hi) percentile bootstrap over examples."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap(
    a: list[float], b: list[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> dict:
    """Paired bootstrap for mean(a) - mean(b) over the *same* eval items.

    Pairing removes item-difficulty variance, which dominates on small sets.
    ``p_value`` is two-sided: 2 * min(P(delta <= 0), P(delta >= 0)).
    """
    x, y = np.asarray(a, float), np.asarray(b, float)
    if x.shape != y.shape:
        raise ValueError("paired bootstrap needs equal-length aligned arrays")
    d = x - y
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    deltas = d[idx].mean(1)
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return {
        "delta": float(d.mean()),
        "lo": float(np.quantile(deltas, alpha / 2)),
        "hi": float(np.quantile(deltas, 1 - alpha / 2)),
        "p_value": float(min(1.0, p)),
        "n": int(d.size),
    }
