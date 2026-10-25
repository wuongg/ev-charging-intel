"""P3 with declared departures versus P4, on the frozen declared-input bank."""

from __future__ import annotations

import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import polars as pl
import yaml

from eci.config import REPO_ROOT, provenance
from eci.data.manifest import sha256_file
from eci.eval.reproduce import reproduce_pscc_v2
from eci.eval.sweep import declared_gap_scenario

LITERATURE_GAP = 0.092


def _mean_gap(rows: list[dict], key: str = "gap") -> float:
    values = [row[key] for row in rows if row.get("n_sessions", 1)]
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def main() -> int:
    bank = REPO_ROOT / "scenarios" / "bank_declared_K50.parquet"
    if not bank.exists():
        print(f"missing {bank}; run scripts/build_declared_bank.py", file=sys.stderr)
        return 1
    protocol = yaml.safe_load((REPO_ROOT / "configs" / "protocol.yaml").read_text(encoding="utf-8"))
    recorded = str(protocol.get("bank_declared_hash", ""))
    actual = sha256_file(bank)
    if recorded != actual:
        print(f"bank_declared_hash {recorded} != file {actual}", file=sys.stderr)
        return 1
    prov = provenance(REPO_ROOT / "configs" / "scheduler.yaml")
    ids = pl.read_parquet(bank).select("scenario_id").unique().sort("scenario_id")["scenario_id"].to_list()
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(declared_gap_scenario, int(scenario_id)) for scenario_id in ids]
        for done, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            print(f"repro_done={done}/{len(ids)}", flush=True)
    gap = _mean_gap(rows)
    energy_gap = _mean_gap(rows, "gap_energy_penalty")
    unmet = _mean_gap(rows, "unmet_p3")
    table = [reproduce_pscc_v2([row["gap"] for row in rows], m_headline=float(rows[0]["M"]), prov=prov)]
    if not (0.09 <= gap <= 0.15):
        table.append(
            {
                "metric": "ablation_demand_off",
                "literature_value": LITERATURE_GAP,
                "repo_value": energy_gap,
                "delta_pp": (energy_gap - LITERATURE_GAP) * 100.0,
                "window": "bank_declared_K50_2018",
                "tariff": "sce_tou_ev_4_march_2019",
                "M": rows[0]["M"],
                "notes": "Same runs with the demand charge left off the bill.",
                **prov,
            }
        )
        for name, months in (("winter", {10, 11, 12, 1, 2, 3, 4, 5}), ("summer", {6, 7, 8, 9})):
            seasonal: list[dict] = []
            with ProcessPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(declared_gap_scenario, int(scenario_id), months) for scenario_id in ids]
                for future in as_completed(futures):
                    seasonal.append(future.result())
            seasonal_gap = _mean_gap(seasonal)
            table.append(
                {
                    "metric": f"ablation_{name}_months",
                    "literature_value": LITERATURE_GAP,
                    "repo_value": seasonal_gap,
                    "delta_pp": (seasonal_gap - LITERATURE_GAP) * 100.0,
                    "window": "bank_declared_K50_2018",
                    "tariff": "sce_tou_ev_4_march_2019",
                    "M": rows[0]["M"],
                    "notes": f"Sessions whose Los Angeles connection month is in the {name} set.",
                    **prov,
                }
            )
    detail = pl.DataFrame(rows)
    detail.write_csv(REPO_ROOT / "results" / "reproduction_v2_detail.csv")
    dest = REPO_ROOT / "results" / "reproduction_v2.csv"
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"gap={gap:.4f} unmet_p3={unmet:.4f} wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
