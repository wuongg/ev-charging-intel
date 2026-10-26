"""Saltelli first-order Sobol on P3 vs P0. N=32, two frozen-bank years."""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import polars as pl

from eci.config import REPO_ROOT, provenance
from eci.eval.sobol import (
    SOBOL_N,
    SOBOL_SCENARIOS,
    sample_matrix,
    sobol_cell,
    sobol_indices,
    spearman_ranking,
)


def main() -> int:
    matrix = sample_matrix(SOBOL_N)
    tasks = []
    if matrix is None:
        print("SALib missing; Spearman fallback on a 16-point grid", flush=True)
        grid = (0.5, 1.0, 2.0)
        m_grid = (10.0, 25.0, 100.0)
        sample_index = 0
        for util in grid:
            for dwell in grid:
                for spread in grid:
                    for penalty in m_grid:
                        for scenario_id in SOBOL_SCENARIOS:
                            tasks.append((sample_index, scenario_id, util, dwell, spread, penalty))
                        sample_index += 1
    else:
        for sample_index, row in enumerate(matrix):
            util, dwell, spread, penalty = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
            for scenario_id in SOBOL_SCENARIOS:
                tasks.append((sample_index, scenario_id, util, dwell, spread, penalty))
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(sobol_cell, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            print(f"sobol={done}/{len(tasks)}", flush=True)
    detail = pl.DataFrame(rows)
    detail.write_csv(REPO_ROOT / "results" / "sobol_detail.csv")
    indices = sobol_indices(detail, SOBOL_N) if matrix is not None else spearman_ranking(detail)
    prov = provenance(REPO_ROOT / "configs" / "scheduler.yaml")
    out = pl.DataFrame(indices).with_columns([pl.lit(prov[key]).alias(key) for key in prov])
    dest = REPO_ROOT / "results" / "sobol_indices.csv"
    out.write_csv(dest)
    print(f"wrote {dest} method={indices[0]['method']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
