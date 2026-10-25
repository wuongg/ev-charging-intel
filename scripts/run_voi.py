"""Deadline VOI on the frozen bank, plus the declared-bank information split."""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import polars as pl

from eci.config import REPO_ROOT, provenance
from eci.eval.stats import paired_cost_test
from eci.eval.voi import FLEET_VEHICLES, SESSIONS_PER_VEHICLE_YEAR, voi_ablation_scenario, voi_scenario


def _ci(frame: pl.DataFrame, group: list[str]) -> pl.DataFrame:
    grouped = frame.group_by(group).agg(
        pl.col("voi").mean().alias("mean_voi"),
        pl.col("voi").std().alias("std_voi"),
        pl.col("voi_fraction_of_p0").mean().alias("mean_fraction"),
        pl.col("n_sessions").mean().alias("mean_sessions"),
        pl.len().alias("n"),
    )
    return grouped.with_columns(
        (
            pl.col("mean_voi") - 1.96 * pl.col("std_voi").fill_null(0.0) / pl.col("n").cast(pl.Float64).sqrt()
        ).alias("ci_low"),
        (
            pl.col("mean_voi") + 1.96 * pl.col("std_voi").fill_null(0.0) / pl.col("n").cast(pl.Float64).sqrt()
        ).alias("ci_high"),
        (
            pl.col("mean_voi") / pl.col("mean_sessions") * SESSIONS_PER_VEHICLE_YEAR * FLEET_VEHICLES
        ).alias("scaled_5000_vehicles"),
    ).drop("std_voi")


def _tests(frame: pl.DataFrame) -> pl.DataFrame:
    rows = []
    keys = frame.select(["tariff", "M"]).unique().sort(["tariff", "M"])
    for tariff, penalty in keys.iter_rows():
        group = frame.filter((pl.col("tariff") == tariff) & (pl.col("M") == penalty)).sort("scenario_id")
        result = paired_cost_test(group["cost_t3"].to_numpy(), group["cost_t2"].to_numpy())
        rows.append({"tariff": tariff, "M": penalty, "comparison": "T3_cost_lt_T2_cost", **result})
    return pl.DataFrame(rows)


def _run(fn, ids: list[int], label: str) -> list[dict]:
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fn, int(scenario_id)) for scenario_id in ids]
        for done, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            print(f"{label}={done}/{len(ids)}", flush=True)
    return rows


def main() -> int:
    frozen = REPO_ROOT / "scenarios" / "bank_K50.parquet"
    declared = REPO_ROOT / "scenarios" / "bank_declared_K50.parquet"
    if not frozen.exists() or not declared.exists():
        print("missing a bank", file=sys.stderr)
        return 1
    prov = provenance(REPO_ROOT / "configs" / "scheduler.yaml")
    frozen_ids = pl.read_parquet(frozen).select("scenario_id").unique().sort("scenario_id")["scenario_id"].to_list()
    declared_ids = (
        pl.read_parquet(declared).select("scenario_id").unique().sort("scenario_id")["scenario_id"].to_list()
    )
    voi_rows = _run(voi_scenario, frozen_ids, "voi_done")
    ablation_rows = _run(voi_ablation_scenario, declared_ids, "ablate_done")
    voi = pl.DataFrame(voi_rows).with_columns([pl.lit(prov[key]).alias(key) for key in prov])
    ablation = pl.DataFrame(ablation_rows).with_columns([pl.lit(prov[key]).alias(key) for key in prov])
    voi.write_csv(REPO_ROOT / "results" / "voi.csv")
    ablation.write_csv(REPO_ROOT / "results" / "voi_ablation.csv")
    summary = _ci(voi, ["bank", "tariff", "M", "info"]).with_columns(
        [pl.lit(prov[key]).alias(key) for key in prov]
    )
    summary.write_csv(REPO_ROOT / "results" / "voi_summary.csv")
    tests = _tests(voi).with_columns([pl.lit(prov[key]).alias(key) for key in prov])
    tests.write_csv(REPO_ROOT / "results" / "voi_tests.csv")
    part = _ci(ablation, ["tariff", "M", "component"])
    part.write_csv(REPO_ROOT / "results" / "voi_ablation_summary.csv")
    print(f"wrote voi rows={voi.height} ablation rows={ablation.height}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
