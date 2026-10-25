"""VOI after T2C. Beta 0.5 is the headline; {0.25, 0.75} are sensitivity."""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import polars as pl

from eci.config import REPO_ROOT, provenance
from eci.eval.stats import paired_cost_test
from eci.eval.voi import FLEET_VEHICLES, SESSIONS_PER_VEHICLE_YEAR, voi_t2c_scenario

BETA_GRID = (0.5, 0.25, 0.75)


def _ci(frame: pl.DataFrame) -> pl.DataFrame:
    grouped = frame.group_by(["tariff", "M", "beta", "info"]).agg(
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


def main() -> int:
    bank = REPO_ROOT / "scenarios" / "bank_K50.parquet"
    if not bank.exists():
        print(f"missing {bank}", file=sys.stderr)
        return 1
    prov = provenance(REPO_ROOT / "configs" / "scheduler.yaml")
    ids = pl.read_parquet(bank).select("scenario_id").unique().sort("scenario_id")["scenario_id"].to_list()
    rows: list[dict] = []
    tasks = [(int(scenario_id), beta) for beta in BETA_GRID for scenario_id in ids]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(voi_t2c_scenario, scenario_id, beta) for scenario_id, beta in tasks]
        for done, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            print(f"voi_c={done}/{len(tasks)}", flush=True)
    frame = pl.DataFrame(rows).with_columns([pl.lit(prov[key]).alias(key) for key in prov])
    frame.write_csv(REPO_ROOT / "results" / "voi_corrected.csv")
    summary = _ci(frame).with_columns([pl.lit(prov[key]).alias(key) for key in prov])
    summary.write_csv(REPO_ROOT / "results" / "voi_corrected_summary.csv")
    tests = []
    keys = frame.select(["tariff", "M", "beta"]).unique().sort(["tariff", "M", "beta"])
    for tariff, penalty, beta in keys.iter_rows():
        group = frame.filter(
            (pl.col("tariff") == tariff) & (pl.col("M") == penalty) & (pl.col("beta") == beta)
        ).sort("scenario_id")
        result = paired_cost_test(group["cost_t3"].to_numpy(), group["cost_t2"].to_numpy())
        tests.append({"tariff": tariff, "M": penalty, "beta": beta, **result, **prov})
    pl.DataFrame(tests).write_csv(REPO_ROOT / "results" / "voi_corrected_tests.csv")
    print(f"wrote voi_corrected rows={frame.height}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
