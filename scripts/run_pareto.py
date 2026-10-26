"""ε-peak sweep on 20-day windows. Writes pareto.csv and a one-day DLMP table."""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import polars as pl

from eci.config import REPO_ROOT, provenance
from eci.eval.pareto import (
    DLMP_EPSILON,
    EPSILON_GRID,
    PARETO_SCENARIOS,
    UTIL_GRID,
    dlmp_day,
    epsilon_sweep_cell,
)


def main() -> int:
    tasks = [(scenario_id, util, eps) for util in UTIL_GRID for eps in EPSILON_GRID for scenario_id in PARETO_SCENARIOS]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(epsilon_sweep_cell, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            print(f"pareto={done}/{len(tasks)}", flush=True)
    prov = provenance(REPO_ROOT / "configs" / "scheduler.yaml")
    frame = pl.DataFrame(rows).with_columns([pl.lit(prov[key]).alias(key) for key in prov])
    dest = REPO_ROOT / "results" / "pareto.csv"
    frame.write_csv(dest)
    dlmp = pl.DataFrame(dlmp_day(0))
    if dlmp.height:
        bind = dlmp.filter(pl.col("cap_dual").abs() > 1e-9)
        shadow = float(bind["cap_dual"].abs().mean()) if bind.height else 0.0
        summary = pl.DataFrame(
            [
                {
                    "metric": "mean_abs_cap_dual_binding",
                    "epsilon": DLMP_EPSILON,
                    "value": shadow,
                    "n_binding": bind.height,
                    "n_slots": dlmp.height,
                    "day": str(dlmp["day"][0]),
                    **prov,
                }
            ]
        )
        summary.write_csv(REPO_ROOT / "results" / "dlmp_summary.csv")
        dlmp.write_csv(REPO_ROOT / "results" / "dlmp_day.csv")
        print(f"shadow={shadow} binding={bind.height}")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
