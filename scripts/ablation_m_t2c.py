"""M sweep on already-solved T2C schedules. Fast filler does not depend on M."""

from __future__ import annotations

import csv
import sys

import polars as pl

from eci.config import REPO_ROOT, provenance
from eci.eval.reproduce import LITERATURE_GAP

M_SWEEP = (10.0, 25.0, 100.0, 1000.0)


def _gap_at_m(row: dict, headline_m: float, penalty: float) -> float:
    unmet_p3 = float(row["penalty_share_p3"]) * float(row["cost_p3"]) / headline_m
    unmet_p4 = float(row["penalty_share_p4"]) * float(row["cost_p4"]) / headline_m
    energy_demand_p3 = float(row["cost_p3"]) - unmet_p3 * headline_m
    energy_demand_p4 = float(row["cost_p4"]) - unmet_p4 * headline_m
    cost_p3 = energy_demand_p3 + unmet_p3 * penalty
    cost_p4 = energy_demand_p4 + unmet_p4 * penalty
    if cost_p4 == 0:
        return float("nan")
    return (cost_p3 - cost_p4) / cost_p4


def main() -> int:
    detail_path = REPO_ROOT / "results" / "reproduction_v4_detail.csv"
    if not detail_path.exists():
        print(f"missing {detail_path}", file=sys.stderr)
        return 1
    detail = pl.read_csv(detail_path)
    headline_m = float(detail["M"][0])
    prov = provenance(REPO_ROOT / "configs" / "scheduler.yaml")
    rows = []
    for penalty in M_SWEEP:
        gaps = [_gap_at_m(row, headline_m, penalty) for row in detail.iter_rows(named=True)]
        mean = float(sum(gaps) / len(gaps))
        rows.append(
            {
                "metric": f"ablation_m_{penalty:g}_t2c",
                "literature_value": LITERATURE_GAP,
                "repo_value": mean,
                "delta_pp": (mean - LITERATURE_GAP) * 100.0,
                "window": "bank_declared_K50_2018",
                "tariff": "sce_tou_ev_4_march_2019",
                "M": penalty,
                "beta": 0.5,
                "notes": (
                    "Same T2C and T3 power as headline M. Bills are re-scored at this "
                    "penalty. Attribution: the gap is unmet kWh times M, not a different schedule."
                ),
                **prov,
            }
        )
    dest = REPO_ROOT / "results" / "reproduction_v4_m_sweep.csv"
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    archive = REPO_ROOT / "results" / "reproduction.csv"
    if archive.exists():
        old = list(csv.DictReader(archive.open(encoding="utf-8")))
        keys = list(dict.fromkeys([*old[0].keys(), *rows[0].keys()])) if old else list(rows[0].keys())
        merged = [{key: row.get(key, "") for key in keys} for row in old]
        seen = {row.get("metric") for row in old}
        merged.extend({key: row.get(key, "") for key in keys} for row in rows if row["metric"] not in seen)
        with archive.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(merged)
    v4 = REPO_ROOT / "results" / "reproduction_v4.csv"
    if v4.exists():
        old = list(csv.DictReader(v4.open(encoding="utf-8")))
        keys = list(dict.fromkeys([*old[0].keys(), *rows[0].keys()])) if old else list(rows[0].keys())
        seen = {row.get("metric") for row in old}
        merged = [{key: row.get(key, "") for key in keys} for row in old]
        merged.extend({key: row.get(key, "") for key in keys} for row in rows if row["metric"] not in seen)
        with v4.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(merged)
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
