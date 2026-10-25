"""Equal-weight quantile ensemble and monotone rearrangement.

Inverse-pinball weights are a pre-registered challenger. They are not fit today.
"""

from __future__ import annotations

import numpy as np


def equal_weight_quantiles(members: list[dict[float, np.ndarray]]) -> dict[float, np.ndarray]:
    """Average q10 with q10, q50 with q50, q90 with q90. No cross-level mixing."""
    if not members:
        raise ValueError("members is empty")
    normalized = [
        {float(level): np.asarray(values, dtype=float) for level, values in member.items()} for member in members
    ]
    levels = list(normalized[0])
    for member in normalized[1:]:
        if set(member) != set(levels):
            raise ValueError("members must share quantile levels")
    averaged: dict[float, np.ndarray] = {}
    for level in levels:
        stacked = np.vstack([member[level] for member in normalized])
        averaged[level] = stacked.mean(axis=0)
    return averaged


def rearrange(quantiles: dict[float, np.ndarray]) -> dict[float, np.ndarray]:
    """Monotone rearrangement: sort each row so quantile levels cannot cross."""
    levels = sorted(float(level) for level in quantiles)
    matrix = np.column_stack([np.asarray(quantiles[level], dtype=float) for level in levels])
    ordered = np.sort(matrix, axis=1)
    return {level: ordered[:, index] for index, level in enumerate(levels)}


def inverse_pinball_weights(*_args: object, **_kwargs: object) -> None:
    """Pre-registered challenger. Not implemented on day 2."""
    raise NotImplementedError(
        "inverse-pinball weights are a pre-registered challenger and are not implemented on day 2"
    )
