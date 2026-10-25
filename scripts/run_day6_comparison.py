"""P0–P4 on SCE and EVN at the headline penalty, after the plant fix."""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import polars as pl

from eci.config import REPO_ROOT, provenance
from eci.eval.sweep import comparison_scenario


def main() -> int:
    bank = REPO_ROOT / "scenarios" / "bank_K50.parquet"
    if not bank.exists():
        print(f"missing {bank}", file=sys.stderr)
        return 1
    prov = provenance(REPO_ROOT / "configs" / "scheduler.yaml")
    ids = pl.read_parquet(bank).select("scenario_id").unique().sort("scenario_id")["scenario_id"].to_list()
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(comparison_scenario, int(scenario_id)) for scenario_id in ids]
        for done, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            print(f"cmp_done={done}/{len(ids)}", flush=True)
    frame = pl.DataFrame(rows).with_columns([pl.lit(prov[key]).alias(key) for key in prov])
    dest = REPO_ROOT / "results" / "day6_comparison.csv"
    frame.write_csv(dest)
    print(f"wrote {dest} rows={frame.height}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
