"""Write results/reproduction.csv from results/metrics.csv."""

from __future__ import annotations

import sys

import polars as pl

from eci.config import REPO_ROOT, provenance
from eci.eval.reproduce import write_reproduction


def main() -> int:
    path = REPO_ROOT / "results" / "metrics.csv"
    if not path.exists():
        print(f"missing {path}; run scripts/run_policies.py first", file=sys.stderr)
        return 1
    frame = pl.read_csv(path, infer_schema_length=None).with_columns(pl.col("value").cast(pl.Float64))
    write_reproduction(frame, provenance(REPO_ROOT / "configs" / "scheduler.yaml"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
