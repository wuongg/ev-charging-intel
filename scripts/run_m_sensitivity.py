"""P0–P4 at four unmet penalties on the frozen bank, SCE, T2v2, cap 150 kW."""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import polars as pl

from eci.config import REPO_ROOT, provenance
from eci.eval.sweep import m_sensitivity_scenario


def _aggregate(rows: list[dict]) -> pl.DataFrame:
    frame = pl.DataFrame(rows)
    grouped = frame.group_by(["M", "policy"]).agg(
        pl.col("reduction_vs_p0").mean().alias("mean_reduction_vs_p0"),
        pl.col("reduction_vs_p0").std().alias("std_reduction"),
        pl.col("penalty_share").mean().alias("penalty_share_of_bill"),
        pl.col("unmet_rate").mean().alias("mean_unmet_rate"),
        pl.len().alias("n"),
    )
    return (
        grouped.with_columns(
            (
                pl.col("mean_reduction_vs_p0")
                - 1.96 * pl.col("std_reduction").fill_null(0.0) / pl.col("n").cast(pl.Float64).sqrt()
            ).alias("ci_low"),
            (
                pl.col("mean_reduction_vs_p0")
                + 1.96 * pl.col("std_reduction").fill_null(0.0) / pl.col("n").cast(pl.Float64).sqrt()
            ).alias("ci_high"),
        )
        .drop("std_reduction")
        .sort(["M", "policy"])
    )


def main() -> int:
    bank = REPO_ROOT / "scenarios" / "bank_K50.parquet"
    if not bank.exists():
        print(f"missing {bank}", file=sys.stderr)
        return 1
    prov = provenance(REPO_ROOT / "configs" / "scheduler.yaml")
    ids = pl.read_parquet(bank).select("scenario_id").unique().sort("scenario_id")["scenario_id"].to_list()
    rows: list[dict] = []
    diagnoses: list[dict] = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(m_sensitivity_scenario, int(scenario_id)) for scenario_id in ids]
        for done, future in enumerate(as_completed(futures), start=1):
            payload = future.result()
            rows.extend(payload["rows"])
            diagnoses.append(payload["diagnosis"])
            print(f"m_done={done}/{len(ids)}", flush=True)
    summary = _aggregate(rows)
    dest = REPO_ROOT / "results" / "m_sensitivity.csv"
    summary.write_csv(dest)
    detail = pl.DataFrame(rows).with_columns([pl.lit(prov[key]).alias(key) for key in prov])
    detail.write_csv(REPO_ROOT / "results" / "m_sensitivity_detail.csv")
    diagnosis = pl.DataFrame(diagnoses).with_columns([pl.lit(prov[key]).alias(key) for key in prov])
    unmet_p3 = float(diagnosis["unmet_p3"].mean())
    if unmet_p3 > 0.10:
        diagnosis_path = REPO_ROOT / "results" / "t2v2_diagnosis.csv"
        diagnosis.write_csv(diagnosis_path)
        print(f"unmet_p3={unmet_p3:.4f} wrote {diagnosis_path}")
    for row in summary.iter_rows(named=True):
        print(
            f"M={row['M']} {row['policy']} red={row['mean_reduction_vs_p0']:.4f} "
            f"share={row['penalty_share_of_bill']:.3f} unmet={row['mean_unmet_rate']:.4f}"
        )
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
