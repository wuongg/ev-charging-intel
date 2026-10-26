"""Dwell × tariff-spread map. Scenario analysis, not a measurement."""

from __future__ import annotations

import polars as pl

from eci.eval.scale import SCALE_NOTE, scale_dwell, scale_spread
from eci.eval.sweep import _fast, _load, _prepare, _sce, _score, headline_penalty
from eci.optimize.policies import charge_on_arrival, cheapest_slots_greedy

DWELL_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
SPREAD_GRID = (0.5, 1.0, 2.0, 4.0)
MAP_SCENARIOS = tuple(range(10))
BREAK_EVEN_PP = 0.02


def dwell_tariff_cell(task: tuple[int, float, float]) -> list[dict]:
    scenario_id, dwell_scale, spread = task
    penalty = headline_penalty()
    fleet = scale_dwell(_prepare(_load("bank_K50.parquet", scenario_id), util=1, impute=True), dwell_scale)
    windows, demand = _sce()
    windows = scale_spread(windows, spread)
    policies = (
        ("P0", charge_on_arrival, "T2"),
        ("P1", cheapest_slots_greedy, "T2"),
        ("P3", _fast(penalty), "T2"),
        ("P4", _fast(penalty), "T3"),
    )
    solved = {}
    for name, policy, tier in policies:
        solved[name] = _score(fleet, policy, tier, name, 150.0, penalty, windows, demand)
    baseline = solved["P0"]["total_cost"]
    energy_base = solved["P0"]["energy_cost"]
    rows = []
    for name, result in solved.items():
        rows.append(
            {
                "scenario_id": scenario_id,
                "dwell_scale": dwell_scale,
                "tariff_spread": spread,
                "policy": name,
                "total_cost": result["total_cost"],
                "energy_cost": result["energy_cost"],
                "unmet_rate": result["unmet_rate"],
                "peak_kw": result["peak_kw"],
                "reduction_vs_p0": ((baseline - result["total_cost"]) / baseline) if baseline else 0.0,
                "energy_reduction_vs_p0": (
                    ((energy_base - result["energy_cost"]) / energy_base) if energy_base else 0.0
                ),
                "voi_p3_minus_p4": solved["P3"]["total_cost"] - solved["P4"]["total_cost"],
                "note": SCALE_NOTE,
            }
        )
    return rows


def median_dwell_hours(scenario_id: int = 0) -> float:
    fleet = _prepare(_load("bank_K50.parquet", scenario_id), util=1, impute=True)
    hours = sorted((session.disconnect - session.connection).total_seconds() / 3600.0 for session in fleet)
    if not hours:
        return float("nan")
    return float(hours[len(hours) // 2])


def dwell_tariff_map(frame: pl.DataFrame, median_hours: float) -> pl.DataFrame:
    """One row per (dwell_scale, tariff_spread) with mean P3 metrics."""
    p3 = frame.filter(pl.col("policy") == "P3")
    return (
        p3.group_by(["dwell_scale", "tariff_spread"])
        .agg(
            pl.col("reduction_vs_p0").mean().alias("mean_reduction_vs_p0"),
            pl.col("energy_reduction_vs_p0").mean().alias("mean_energy_reduction_vs_p0"),
            pl.col("voi_p3_minus_p4").mean().alias("mean_voi"),
            pl.col("unmet_rate").mean().alias("mean_unmet"),
            pl.len().alias("n"),
        )
        .with_columns(
            (pl.col("dwell_scale") * median_hours).alias("median_dwell_h"),
            pl.lit(SCALE_NOTE).alias("note"),
        )
        .sort(["tariff_spread", "dwell_scale"])
    )


def break_even_hours(summary: pl.DataFrame, median_hours: float, column: str = "mean_reduction_vs_p0") -> dict:
    """First dwell_scale at spread=1 whose mean savings exceed 2pp. Scenario analysis."""
    arm = summary.filter(pl.col("tariff_spread") == 1.0).sort("dwell_scale")
    chosen = None
    used = column
    below_scale = None
    previous = None
    for row in arm.iter_rows(named=True):
        if float(row[column]) > BREAK_EVEN_PP:
            chosen = row
            below_scale = previous["dwell_scale"] if previous is not None else row["dwell_scale"]
            break
        previous = row
    if chosen is None:
        used = "mean_energy_reduction_vs_p0"
        previous = None
        for row in arm.iter_rows(named=True):
            if float(row[used]) > BREAK_EVEN_PP:
                chosen = row
                below_scale = previous["dwell_scale"] if previous is not None else row["dwell_scale"]
                break
            previous = row
    return {
        "metric": used,
        "dwell_scale": float(chosen["dwell_scale"]) if chosen else float("nan"),
        "median_dwell_h": float(chosen["dwell_scale"]) * median_hours if chosen else float("nan"),
        "below_h": float(below_scale) * median_hours if below_scale is not None else float("nan"),
        "found": chosen is not None,
        "base_median_h": median_hours,
    }


def monotonicity_flags(summary: pl.DataFrame, column: str = "mean_reduction_vs_p0") -> list[dict]:
    """Flag (do not fail) a spread arm whose savings fall when dwell_scale rises."""
    flags = []
    for spread in sorted(summary["tariff_spread"].unique().to_list()):
        arm = summary.filter(pl.col("tariff_spread") == spread).sort("dwell_scale")
        scales = arm["dwell_scale"].to_list()
        values = arm[column].to_list()
        for left, right, v_left, v_right in zip(scales, scales[1:], values, values[1:], strict=False):
            if float(v_right) + 1e-6 < float(v_left):
                flags.append(
                    {
                        "tariff_spread": spread,
                        "dwell_scale_from": left,
                        "dwell_scale_to": right,
                        "value_from": v_left,
                        "value_to": v_right,
                        "column": column,
                    }
                )
    return flags
