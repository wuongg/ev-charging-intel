"""P0, P1, P2b, P3, and the T3 oracle across the frozen K=50 bank and both tariffs."""

from __future__ import annotations

import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta

import polars as pl
import yaml

from eci.config import REPO_ROOT, load_config, provenance
from eci.eval.dispatch import declared_inputs, evaluate_scenario, sessions_from_rows
from eci.eval.stats import benjamini_hochberg, paired_cost_test

METRICS = (
    "total_cost",
    "energy_cost",
    "penalty_cost",
    "demand_cost",
    "cost_reduction_vs_p0",
    "optimality_gap_p3_p4",
    "gap_energy_only",
    "gap_deadline_inside",
    "unmet_rate",
    "peak_kw",
    "plug_binding_fraction",
    "solve_time_p95",
    "min_cap_kw",
)


def _tariffs() -> list[dict]:
    sce = load_config(REPO_ROOT / "configs" / "tariff_sce_ev4.yaml").model_dump()
    evn = load_config(REPO_ROOT / "configs" / "tariff_evn_2026.yaml").model_dump()
    two_part = yaml.safe_load((REPO_ROOT / "configs" / "tariff_evn_twopart.yaml").read_text(encoding="utf-8"))
    return [
        {
            "name": "sce_ev4",
            "windows": sce["windows"],
            "demand_rate": float(sce["demand_charge_per_kw_month"]),
        },
        {
            "name": "evn_2026",
            "windows": evn["windows"],
            "demand_rate": float(two_part["capacity_charge_vnd_per_kw_month"]["kv22_to_110"]),
        },
    ]


def _median_dwell(path) -> timedelta:
    if not path.exists():
        return timedelta(hours=4.53)
    frame = pl.read_parquet(path).filter(
        (pl.col("siteID") == "0002")
        & (pl.col("kWhDelivered") > 0)
        & (pl.col("connectionTime_utc").dt.year() == 2019)
    )
    hours = (pl.col("disconnectTime_utc") - pl.col("connectionTime_utc")).dt.total_seconds() / 3600.0
    value = frame.select(hours.alias("dwell_h")).filter(pl.col("dwell_h") > 0)["dwell_h"].median()
    return timedelta(hours=float(value or 4.53))


_DECLARED: dict | None = None


def _declared() -> dict:
    global _DECLARED
    if _DECLARED is None:
        _DECLARED = declared_inputs(REPO_ROOT / "data" / "interim" / "sessions.parquet")
    return _DECLARED


def _run_one(task: tuple) -> list[dict]:
    scenario_id, median_hours, cap_kw, volts, default_kw, penalty = task
    bank = pl.read_parquet(REPO_ROOT / "scenarios" / "bank_K50.parquet")
    declared = _declared()
    rows = bank.filter(pl.col("scenario_id") == scenario_id).to_dicts()
    sessions = sessions_from_rows(rows, volts, default_kw, declared)
    evaluated = evaluate_scenario(
        sessions,
        cap_kw=cap_kw,
        tariffs=_tariffs(),
        median_dwell=timedelta(hours=median_hours),
        penalty=penalty,
    )
    for row in evaluated:
        row["scenario_id"] = scenario_id
    return evaluated


def _metric_rows(evaluated: list[dict], note: str, prov: dict) -> list[dict]:
    flat = []
    for row in evaluated:
        for metric in METRICS:
            flat.append(
                {
                    "tariff": row["tariff"],
                    "policy": row["policy"],
                    "scenario_id": row["scenario_id"],
                    "metric": metric,
                    "value": row[metric],
                    "tier": row["tier"],
                    "note": note,
                    **prov,
                }
            )
    return flat


def _family(frame: pl.DataFrame) -> list[dict]:
    tests = []
    comparisons = (
        ("sce_ev4", "P3", "P1", True),
        ("sce_ev4", "P3", "P0", False),
        ("sce_ev4", "P1", "P0", False),
        ("sce_ev4", "P2b", "P0", False),
        ("evn_2026", "P3", "P1", False),
        ("evn_2026", "P3", "P0", False),
        ("evn_2026", "P1", "P0", False),
        ("evn_2026", "P2b", "P0", False),
    )
    costs = frame.filter(pl.col("metric") == "total_cost")
    for tariff, left, right, primary in comparisons:
        a = costs.filter((pl.col("tariff") == tariff) & (pl.col("policy") == left)).sort("scenario_id")
        b = costs.filter((pl.col("tariff") == tariff) & (pl.col("policy") == right)).sort("scenario_id")
        result = paired_cost_test(a["value"].to_numpy(), b["value"].to_numpy())
        tests.append(
            {
                "tariff": tariff,
                "comparison": f"{left}_vs_{right}",
                "primary": primary,
                **result,
            }
        )
    adjusted = benjamini_hochberg([row["p_value"] for row in tests])
    for row, p_adj in zip(tests, adjusted, strict=True):
        row["p_bh"] = p_adj
        row["reject_q10"] = p_adj <= 0.10
    return tests


def _summary(frame: pl.DataFrame) -> pl.DataFrame:
    reduction = frame.filter((pl.col("metric") == "cost_reduction_vs_p0") & (pl.col("policy") != "P0"))
    grouped = reduction.group_by(["tariff", "policy"]).agg(
        pl.col("value").mean().alias("mean_cost_reduction_vs_p0"),
        pl.col("value").std().alias("std"),
        pl.len().alias("n"),
    )
    return (
        grouped.with_columns(
            (
                pl.col("mean_cost_reduction_vs_p0")
                - 1.96 * pl.col("std") / pl.col("n").cast(pl.Float64).sqrt()
            ).alias("ci_low"),
            (
                pl.col("mean_cost_reduction_vs_p0")
                + 1.96 * pl.col("std") / pl.col("n").cast(pl.Float64).sqrt()
            ).alias("ci_high"),
        )
        .drop("std")
        .sort(["tariff", "policy"])
    )


def main() -> int:
    if "--m-sensitivity" in sys.argv:
        from run_m_sensitivity import main as run_m

        return run_m()
    if "--cap-sweep" in sys.argv:
        from run_cap_sweep import main as run_cap

        return run_cap()
    bank_path = REPO_ROOT / "scenarios" / "bank_K50.parquet"
    if not bank_path.exists():
        print(f"missing {bank_path}", file=sys.stderr)
        return 1
    prov = provenance(REPO_ROOT / "configs" / "scheduler.yaml")
    site = load_config(REPO_ROOT / "configs" / "site_caltech.yaml").model_dump()
    scheduler = load_config(REPO_ROOT / "configs" / "scheduler.yaml").model_dump()
    median = _median_dwell(REPO_ROOT / "data" / "interim" / "sessions.parquet")
    ids = pl.read_parquet(bank_path).select("scenario_id").unique().sort("scenario_id")["scenario_id"].to_list()
    tasks = [
        (
            int(scenario_id),
            median.total_seconds() / 3600.0,
            float(site["site_cap_kw"]),
            208.0,
            6.7,
            float(scheduler["penalty_unmet"]),
        )
        for scenario_id in ids
    ]
    note = (
        "Windows applied on the America/Los_Angeles clock of the frozen bank. "
        "P0/P1/P2b/P3 use T2: declared userInputs when the session has one, otherwise "
        "deadline = connection + median 2019 Caltech dwell and energy = delivered kWh. "
        "P4 is the T3 oracle. The frozen bank has no userInputs column; the lookup is the "
        "source session, not a new draw. "
        "Bill = energy + M*unmet + monthly demand charge. "
        "kWh==0 sessions are absent from this bank. "
        "EVN prices are VND and are not converted. The frozen M stays 1000 in each tariff's unit. "
    )
    evaluated: list[dict] = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_run_one, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), start=1):
            batch = future.result()
            evaluated.extend(batch)
            print(f"scenario_done={done}/{len(tasks)} id={batch[0]['scenario_id']}", flush=True)
    flat = _metric_rows(evaluated, note, prov)
    dest = REPO_ROOT / "results" / "metrics.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    frame = pl.read_csv(dest, infer_schema_length=None).with_columns(pl.col("value").cast(pl.Float64))
    summary = _summary(frame)
    summary_path = REPO_ROOT / "results" / "metrics_summary.csv"
    summary.write_csv(summary_path)
    print(summary)
    tests = _family(frame)
    inference = REPO_ROOT / "results" / "inference.csv"
    with inference.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tests[0]))
        writer.writeheader()
        writer.writerows(tests)
    for row in tests:
        if row["primary"]:
            print(
                f"primary {row['comparison']} {row['tariff']} p={row['p_value']:.4g} "
                f"p_bh={row['p_bh']:.4g} reject={row['reject_q10']}"
            )
    from eci.eval.reproduce import write_reproduction

    write_reproduction(frame, prov)
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
