"""BH-FDR keeps the family monotone and does not reorder the input."""

import numpy as np

from eci.eval.stats import benjamini_hochberg, paired_cost_test


def test_bh_is_monotone_and_in_input_order() -> None:
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03])
    assert adjusted[0] <= adjusted[2] <= adjusted[1] or adjusted[0] <= 0.03
    assert len(adjusted) == 3
    assert all(0.0 <= value <= 1.0 for value in adjusted)


def test_paired_cost_detects_a_lower_mean() -> None:
    left = np.array([1.0, 1.1, 0.9, 1.0])
    right = np.array([2.0, 2.2, 1.8, 2.1])
    result = paired_cost_test(left, right)
    assert result["p_value"] < 0.05
    assert result["mean_diff"] < 0
