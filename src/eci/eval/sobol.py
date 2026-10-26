"""First-order Sobol indices on P3-vs-P0 reduction. Scenario analysis."""

from __future__ import annotations

import numpy as np
import polars as pl

from eci.eval.scale import SCALE_NOTE, apply_util, scale_dwell, scale_spread
from eci.eval.sweep import _fast, _load, _prepare, _sce, _score
from eci.optimize.policies import charge_on_arrival

FACTORS = ("utilization", "dwell_scale", "tariff_spread", "M")
BOUNDS = {
    "utilization": (0.5, 2.0),
    "dwell_scale": (0.5, 2.0),
    "tariff_spread": (0.5, 2.0),
    "M": (10.0, 100.0),
}
SOBOL_N = 32
SOBOL_SCENARIOS = (0, 1)


def problem() -> dict:
    return {
        "num_vars": 4,
        "names": list(FACTORS),
        "bounds": [list(BOUNDS[name]) for name in FACTORS],
    }


def sample_matrix(n: int = SOBOL_N, seed: int = 7):
    try:
        from SALib.sample import saltelli
    except ImportError:
        return None
    del seed
    return saltelli.sample(problem(), n, calc_second_order=False, skip_values=1024)


def sobol_cell(task: tuple[int, int, float, float, float, float]) -> dict:
    sample_index, scenario_id, util, dwell, spread, penalty = task
    fleet = scale_dwell(
        apply_util(_prepare(_load("bank_K50.parquet", scenario_id), util=1, impute=True), util),
        dwell,
    )
    windows, demand = _sce()
    windows = scale_spread(windows, spread)
    p0 = _score(fleet, charge_on_arrival, "T2", "P0", 150.0, penalty, windows, demand)
    p3 = _score(fleet, _fast(penalty), "T2", "P3", 150.0, penalty, windows, demand)
    base = p0["total_cost"]
    reduction = ((base - p3["total_cost"]) / base) if base else 0.0
    return {
        "sample_index": sample_index,
        "scenario_id": scenario_id,
        "utilization": util,
        "dwell_scale": dwell,
        "tariff_spread": spread,
        "M": penalty,
        "reduction_vs_p0": reduction,
        "note": SCALE_NOTE,
    }


def mean_y(frame: pl.DataFrame) -> np.ndarray:
    return (
        frame.group_by("sample_index")
        .agg(pl.col("reduction_vs_p0").mean())
        .sort("sample_index")["reduction_vs_p0"]
        .to_numpy()
    )


def sobol_indices(frame: pl.DataFrame, n: int = SOBOL_N) -> list[dict]:
    y = mean_y(frame)
    try:
        from SALib.analyze import sobol as analyze
    except ImportError:
        return spearman_ranking(frame)
    si = analyze.analyze(problem(), y, calc_second_order=False, print_to_console=False)
    rows = []
    for index, name in enumerate(FACTORS):
        rows.append(
            {
                "factor": name,
                "S1": float(si["S1"][index]),
                "S1_conf": float(si["S1_conf"][index]),
                "ST": float(si["ST"][index]),
                "ST_conf": float(si["ST_conf"][index]),
                "n": n,
                "n_scenarios": len(SOBOL_SCENARIOS),
                "method": "saltelli",
                "note": SCALE_NOTE,
            }
        )
    return rows


def spearman_ranking(frame: pl.DataFrame) -> list[dict]:
    from scipy.stats import spearmanr

    y = frame["reduction_vs_p0"].to_numpy()
    rows = []
    for name in FACTORS:
        x = frame[name].to_numpy()
        if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
            rho = 0.0
        else:
            rho, _p = spearmanr(x, y)
        value = float(rho) if rho == rho else 0.0
        rows.append(
            {
                "factor": name,
                "S1": value,
                "S1_conf": float("nan"),
                "ST": abs(value),
                "ST_conf": float("nan"),
                "n": frame.height,
                "n_scenarios": len(SOBOL_SCENARIOS),
                "method": "spearman_fallback",
                "note": SCALE_NOTE + " SALib missing; rank correlation only.",
            }
        )
    return rows
