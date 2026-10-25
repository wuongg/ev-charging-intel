"""PSCC gap and the minimum site cap, read off the bank metrics."""

from __future__ import annotations

import csv
from pathlib import Path

import polars as pl

LITERATURE_GAP = 0.092
LITERATURE_CAP_KW = 60.0


def reproduction_table(frame: pl.DataFrame) -> list[dict]:
    """Two headline rows, plus energy-only and a T3 note when the gap is far from 9.2%."""
    sce = frame.filter(pl.col("tariff") == "sce_ev4")
    gaps = sce.filter((pl.col("policy") == "P3") & (pl.col("metric") == "optimality_gap_p3_p4"))["value"].to_numpy()
    energy_gaps = sce.filter((pl.col("policy") == "P3") & (pl.col("metric") == "gap_energy_only"))["value"].to_numpy()
    caps = sce.filter((pl.col("policy") == "P4") & (pl.col("metric") == "min_cap_kw"))["value"].to_numpy()
    gap = float(gaps.mean()) if len(gaps) else float("nan")
    energy_gap = float(energy_gaps.mean()) if len(energy_gaps) else float("nan")
    inside_values = sce.filter((pl.col("policy") == "P3") & (pl.col("metric") == "gap_deadline_inside"))[
        "value"
    ].to_numpy()
    inside = float(inside_values.mean()) if len(inside_values) else float("nan")
    cap_mean = float(caps.mean()) if len(caps) else float("nan")
    cap_max = float(caps.max()) if len(caps) else float("nan")
    rows = [
        {
            "metric": "pscc_gap_p3_t2_vs_p4_t3",
            "literature_value": LITERATURE_GAP,
            "repo_value": gap,
            "delta_pp": (gap - LITERATURE_GAP) * 100.0,
            "window": "bank_K50_calendar_2019",
            "tariff": "sce_tou_ev_4_march_2019",
            "notes": (
                "Mean over scenarios of (cost_P3 - cost_P4) / cost_P4. "
                "Bill includes the 15.51 USD/kW monthly demand charge."
            ),
        },
        {
            "metric": "min_capacity_kw_mean_of_years",
            "literature_value": LITERATURE_CAP_KW,
            "repo_value": cap_mean,
            "delta_pp": cap_mean - LITERATURE_CAP_KW,
            "window": "bank_K50_calendar_2019",
            "tariff": "none_feasibility",
            "notes": (
                "Mean across years of the per-year minimum cap. "
                "delta_pp is kW for this row. The bank-wide max is the next row."
            ),
        },
        {
            "metric": "min_capacity_kw_serves_every_bank_day",
            "literature_value": LITERATURE_CAP_KW,
            "repo_value": cap_max,
            "delta_pp": cap_max - LITERATURE_CAP_KW,
            "window": "bank_K50_calendar_2019",
            "tariff": "none_feasibility",
            "notes": (
                "Maximum of the per-year caps, so one cap serves every bank day. "
                "delta_pp is kW. Undeliverable kWh at an unlimited cap is excluded."
            ),
        },
    ]
    if abs(gap - LITERATURE_GAP) > 0.15:
        rows.append(
            {
                "metric": "ablation_energy_and_penalty_only",
                "literature_value": LITERATURE_GAP,
                "repo_value": energy_gap,
                "delta_pp": (energy_gap - LITERATURE_GAP) * 100.0,
                "window": "bank_K50_calendar_2019",
                "tariff": "sce_tou_ev_4_march_2019",
                "notes": (
                    "Same runs, demand charge removed from the bill. "
                    "This is the demand-charge share of the gap."
                ),
            }
        )
        rows.append(
            {
                "metric": "ablation_deadline_inside_stay",
                "literature_value": LITERATURE_GAP,
                "repo_value": inside,
                "delta_pp": (inside - LITERATURE_GAP) * 100.0,
                "window": "bank_K50_calendar_2019",
                "tariff": "sce_tou_ev_4_march_2019",
                "notes": (
                    "Same P3/P4 pair on sessions whose T2 deadline is no later than "
                    "the true disconnect. Drops early departures the median over-states."
                ),
            }
        )
        rows.append(
            {
                "metric": "ablation_p3_under_t3",
                "literature_value": LITERATURE_GAP,
                "repo_value": 0.0,
                "delta_pp": (0.0 - LITERATURE_GAP) * 100.0,
                "window": "bank_K50_calendar_2019",
                "tariff": "sce_tou_ev_4_march_2019",
                "notes": (
                    "P3 with T3 information is the P4 run, so the gap is 0. "
                    "The headline gap is the imputed deadline, not a different scheduler."
                ),
            }
        )
    return rows


def write_reproduction(frame: pl.DataFrame, prov: dict, dest: Path | None = None) -> Path:
    from eci.config import REPO_ROOT

    dest = dest or (REPO_ROOT / "results" / "reproduction.csv")
    rows = reproduction_table(frame)
    for row in rows:
        row.update(prov)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['metric']} repo={row['repo_value']:.4f} "
            f"literature={row['literature_value']} delta={row['delta_pp']:.2f}"
        )
    print(f"wrote {dest}")
    return dest
