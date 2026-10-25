"""P0–P4 on both tariffs after T2C. Writes results/metrics_v4.csv."""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import polars as pl

from eci.config import REPO_ROOT, provenance
from eci.eval.sweep import comparison_t2c_scenario


def main() -> int:
    bank = REPO_ROOT / "scenarios" / "bank_K50.parquet"
    if not bank.exists():
        print(f"missing {bank}", file=sys.stderr)
        return 1
    prov = provenance(REPO_ROOT / "configs" / "scheduler.yaml")
    ids = pl.read_parquet(bank).select("scenario_id").unique().sort("scenario_id")["scenario_id"].to_list()
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(comparison_t2c_scenario, int(scenario_id)) for scenario_id in ids]
        for done, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            print(f"v4_cmp={done}/{len(ids)}", flush=True)
    frame = pl.DataFrame(rows).with_columns([pl.lit(prov[key]).alias(key) for key in prov])
    dest = REPO_ROOT / "results" / "metrics_v4.csv"
    frame.write_csv(dest)
    summary = (
        frame.group_by(["tariff", "policy"])
        .agg(
            pl.col("reduction_vs_p0").mean().alias("mean_reduction_vs_p0"),
            pl.col("reduction_vs_p0").std().alias("std"),
            pl.col("unmet_rate").mean().alias("mean_unmet_rate"),
            pl.col("penalty_share").mean().alias("penalty_share_of_bill"),
            pl.len().alias("n"),
        )
        .with_columns(
            (
                pl.col("mean_reduction_vs_p0")
                - 1.96 * pl.col("std").fill_null(0.0) / pl.col("n").cast(pl.Float64).sqrt()
            ).alias("ci_low"),
            (
                pl.col("mean_reduction_vs_p0")
                + 1.96 * pl.col("std").fill_null(0.0) / pl.col("n").cast(pl.Float64).sqrt()
            ).alias("ci_high"),
        )
        .drop("std")
        .sort(["tariff", "policy"])
    )
    summary.write_csv(REPO_ROOT / "results" / "metrics_v4_summary.csv")
    print(f"wrote {dest} rows={frame.height}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
