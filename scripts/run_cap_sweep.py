"""Cap × utilization grid for P0, P1, P3, and P4 at 60 kW. Twenty scenarios."""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import polars as pl

from eci.config import REPO_ROOT, provenance
from eci.eval.stats import paired_cost_test
from eci.eval.sweep import cap_sweep_scenario


def _tests(frame: pl.DataFrame) -> pl.DataFrame:
    rows = []
    keys = frame.select(["cap", "util"]).unique().sort(["util", "cap"])
    for cap, util in keys.iter_rows():
        group = frame.filter((pl.col("cap") == cap) & (pl.col("util") == util))
        p1 = group.filter(pl.col("policy") == "P1").sort("scenario_id")["cost"].to_numpy()
        p3 = group.filter(pl.col("policy") == "P3").sort("scenario_id")["cost"].to_numpy()
        result = paired_cost_test(p3, p1)
        rows.append({"cap": cap, "util": util, "comparison": "P3_vs_P1", **result})
    return pl.DataFrame(rows)


def main() -> int:
    bank = REPO_ROOT / "scenarios" / "bank_K50.parquet"
    if not bank.exists():
        print(f"missing {bank}", file=sys.stderr)
        return 1
    prov = provenance(REPO_ROOT / "configs" / "scheduler.yaml")
    ids = pl.read_parquet(bank).select("scenario_id").unique().sort("scenario_id")["scenario_id"].to_list()
    ids = ids[:20]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(cap_sweep_scenario, int(scenario_id)) for scenario_id in ids]
        for done, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            print(f"cap_done={done}/{len(ids)}", flush=True)
    frame = pl.DataFrame(rows).with_columns([pl.lit(prov[key]).alias(key) for key in prov])
    dest = REPO_ROOT / "results" / "cap_sweep.csv"
    frame.write_csv(dest)
    tests = _tests(frame).with_columns([pl.lit(prov[key]).alias(key) for key in prov])
    tests.write_csv(REPO_ROOT / "results" / "cap_sweep_tests.csv")
    print(f"wrote {dest}")
    print(prov)
    return 0


if __name__ == "__main__":
    sys.exit(main())
