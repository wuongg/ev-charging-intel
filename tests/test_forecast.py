"""Pinball, rearrangement, and the pre-registered weight challenger."""

from __future__ import annotations

import numpy as np
import pytest

from eci.forecast.ensemble import equal_weight_quantiles, inverse_pinball_weights, rearrange
from eci.forecast.metrics import empirical_coverage, paired_moving_block_bootstrap, pinball_loss


def test_pinball_known_values() -> None:
    assert pinball_loss(np.array([10.0]), np.array([0.0]), 0.1) == pytest.approx(1.0)
    assert pinball_loss(np.array([0.0]), np.array([10.0]), 0.1) == pytest.approx(9.0)


def test_rearrange_sorts_each_row() -> None:
    ordered = rearrange({0.1: np.array([3.0, 0.0]), 0.5: np.array([1.0, 2.0]), 0.9: np.array([2.0, 1.0])})
    assert ordered[0.1].tolist() == [1.0, 0.0]
    assert ordered[0.5].tolist() == [2.0, 1.0]
    assert ordered[0.9].tolist() == [3.0, 2.0]


def test_equal_weight_averages_matching_levels() -> None:
    averaged = equal_weight_quantiles(
        [
            {0.1: np.array([0.0, 2.0]), 0.9: np.array([4.0, 6.0])},
            {0.1: np.array([2.0, 4.0]), 0.9: np.array([6.0, 8.0])},
        ]
    )
    assert averaged[0.1].tolist() == [1.0, 3.0]
    assert averaged[0.9].tolist() == [5.0, 7.0]


def test_inverse_pinball_weights_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        inverse_pinball_weights([1.0, 2.0])


def test_coverage_nominal_interval() -> None:
    actual = np.array([0.0, 1.0, 2.0, 3.0])
    assert empirical_coverage(actual, np.zeros(4), np.full(4, 3.0)) == 1.0


def test_block_bootstrap_detects_a_lower_series() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=400)
    result = paired_moving_block_bootstrap(base - 1.0, base, block=24, B=200, seed=1)
    assert result["mean_diff"] < 0
    assert result["p_value"] < 0.05
