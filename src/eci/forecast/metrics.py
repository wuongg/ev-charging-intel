"""Pinball, interval coverage, and a paired moving-block bootstrap."""

from __future__ import annotations

import numpy as np


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array)
    return mask


def pinball_loss(actual: np.ndarray, pred_q: np.ndarray, tau: float) -> float:
    """Mean pinball loss at level tau. Non-finite pairs are dropped."""
    actual = np.asarray(actual, dtype=float)
    pred_q = np.asarray(pred_q, dtype=float)
    mask = _finite_mask(actual, pred_q)
    if not mask.any():
        return float("nan")
    error = actual[mask] - pred_q[mask]
    loss = np.where(error >= 0, tau * error, (tau - 1.0) * error)
    return float(np.mean(loss))


def pinball_series(actual: np.ndarray, pred_q: np.ndarray, tau: float) -> np.ndarray:
    """Per-timestamp pinball. Non-finite pairs become NaN so the bootstrap can skip them."""
    actual = np.asarray(actual, dtype=float)
    pred_q = np.asarray(pred_q, dtype=float)
    error = actual - pred_q
    loss = np.where(error >= 0, tau * error, (tau - 1.0) * error)
    loss = loss.astype(float)
    loss[~_finite_mask(actual, pred_q)] = np.nan
    return loss


def empirical_coverage(actual: np.ndarray, q_low: np.ndarray, q_high: np.ndarray) -> float:
    """Share of actuals inside [q_low, q_high]. For q10 and q90 the nominal rate is 80%."""
    actual = np.asarray(actual, dtype=float)
    q_low = np.asarray(q_low, dtype=float)
    q_high = np.asarray(q_high, dtype=float)
    mask = _finite_mask(actual, q_low, q_high)
    if not mask.any():
        return float("nan")
    inside = (actual[mask] >= q_low[mask]) & (actual[mask] <= q_high[mask])
    return float(np.mean(inside))


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    """Mean absolute error. Non-finite pairs are dropped."""
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = _finite_mask(actual, pred)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(actual[mask] - pred[mask])))


def quantile_crossing_rate(q10: np.ndarray, q50: np.ndarray, q90: np.ndarray) -> float:
    """Share of rows where a lower quantile sits above a higher one."""
    q10 = np.asarray(q10, dtype=float)
    q50 = np.asarray(q50, dtype=float)
    q90 = np.asarray(q90, dtype=float)
    mask = _finite_mask(q10, q50, q90)
    if not mask.any():
        return float("nan")
    crossed = (q10[mask] > q50[mask]) | (q50[mask] > q90[mask]) | (q10[mask] > q90[mask])
    return float(np.mean(crossed))


def paired_moving_block_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    block: int = 168,
    B: int = 2000,
    seed: int = 7,
) -> dict[str, float | int]:
    """One-sided test that mean(a − b) < 0, plus a percentile CI for that mean.

    Blocks are circular. The null distribution recenters the bootstrap means at
    0 (H0: the mean difference is 0). A small p-value means series ``a`` is
    smaller than series ``b``. Pass loss series in, so a small p-value means
    member A beat member B.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    diff = diff[np.isfinite(diff)]
    n = int(diff.size)
    empty = {
        "p_value": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "mean_diff": float("nan"),
        "n": 0,
    }
    if n == 0:
        return empty
    block = max(1, min(int(block), n))
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    means = np.empty(B, dtype=float)
    for draw in range(B):
        starts = rng.integers(0, n, size=n_blocks)
        pieces = [diff[(start + np.arange(block)) % n] for start in starts]
        means[draw] = np.concatenate(pieces)[:n].mean()
    observed = float(diff.mean())
    null = means - observed
    p_value = float((1 + np.sum(null <= observed)) / (B + 1))
    return {
        "p_value": p_value,
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "mean_diff": observed,
        "n": n,
    }
