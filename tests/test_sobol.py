"""Sobol problem statement. The bank sweep lives in scripts/run_sobol.py."""

import polars as pl

from eci.eval.sobol import FACTORS, problem, sample_matrix, spearman_ranking


def test_problem_has_four_factors() -> None:
    spec = problem()
    assert spec["num_vars"] == 4
    assert spec["names"] == list(FACTORS)


def test_spearman_fallback_ranks_a_known_slope() -> None:
    frame = pl.DataFrame(
        {
            "utilization": [0.5, 1.0, 2.0, 0.5, 1.0, 2.0],
            "dwell_scale": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "tariff_spread": [0.5, 0.5, 0.5, 2.0, 2.0, 2.0],
            "M": [25.0, 25.0, 25.0, 25.0, 25.0, 25.0],
            "reduction_vs_p0": [0.0, 0.0, 0.0, 0.1, 0.2, 0.3],
        }
    )
    rows = {row["factor"]: row for row in spearman_ranking(frame)}
    assert rows["tariff_spread"]["ST"] > rows["dwell_scale"]["ST"]


def test_saltelli_sample_length_when_salib_present() -> None:
    matrix = sample_matrix(8)
    if matrix is None:
        return
    # N*(D+2) first-order Saltelli.
    assert matrix.shape[0] == 8 * (4 + 2)
    assert matrix.shape[1] == 4
