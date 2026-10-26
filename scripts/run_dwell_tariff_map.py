"""6×4 dwell × spread map on 10 frozen-bank scenarios. Scenario analysis."""

from __future__ import annotations

import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import polars as pl

from eci.config import REPO_ROOT, provenance
from eci.eval.heatmap import (
    DWELL_GRID,
    MAP_SCENARIOS,
    SPREAD_GRID,
    break_even_hours,
    dwell_tariff_cell,
    dwell_tariff_map,
    median_dwell_hours,
    monotonicity_flags,
)


def main() -> int:
    tasks = [(scenario_id, s, r) for s in DWELL_GRID for r in SPREAD_GRID for scenario_id in MAP_SCENARIOS]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(dwell_tariff_cell, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            print(f"map={done}/{len(tasks)}", flush=True)
    detail = pl.DataFrame(rows)
    dest = REPO_ROOT / "results" / "dwell_tariff_map.csv"
    detail.write_csv(dest)
    hours = median_dwell_hours(0)
    summary = dwell_tariff_map(detail, hours)
    summary.write_csv(REPO_ROOT / "results" / "dwell_tariff_map_summary.csv")
    even = break_even_hours(summary, hours)
    flags = monotonicity_flags(summary)
    even["n_flags"] = len(flags)
    even["n_cells"] = summary.height
    even["base_median_h"] = hours
    prov = provenance(REPO_ROOT / "configs" / "scheduler.yaml")
    even.update(prov)
    with (REPO_ROOT / "results" / "dwell_tariff_break_even.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(even))
        writer.writeheader()
        writer.writerow(even)
    if flags:
        pl.DataFrame(flags).write_csv(REPO_ROOT / "results" / "dwell_tariff_monotonicity_flags.csv")
    print(f"wrote {dest} hours={hours:.3f} break_even={even['median_dwell_h']} flags={len(flags)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
