"""Paired tests and Benjamini-Hochberg adjustment at the pre-registered q."""

from __future__ import annotations

import numpy as np
from scipy import stats


def paired_cost_test(a: np.ndarray, b: np.ndarray) -> dict:
    """One-sided test that mean(a) < mean(b), plus a 95% CI for mean(a - b)."""
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if left.shape != right.shape or left.size < 2:
        raise ValueError("paired test needs at least two aligned scenarios")
    diff = left - right
    mean = float(np.mean(diff))
    if float(np.std(diff)) == 0.0:
        p_value = 0.0 if mean < 0 else 1.0
        se = 0.0
    else:
        result = stats.ttest_rel(left, right, alternative="less")
        p_value = float(result.pvalue)
        se = float(np.std(diff, ddof=1) / np.sqrt(diff.size))
    half = 1.96 * se
    return {
        "p_value": p_value,
        "mean_diff": mean,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "n": int(diff.size),
    }


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Adjusted p-values, monotone from the largest rank down."""
    values = np.asarray(p_values, dtype=float)
    m = values.size
    if m == 0:
        return []
    order = np.argsort(values)
    adjusted = np.empty(m, dtype=float)
    running = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        running = min(running, float(values[idx]) * m / rank)
        adjusted[idx] = running
    return [float(min(1.0, item)) for item in adjusted]
