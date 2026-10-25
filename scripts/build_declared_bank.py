"""Draw the declared-departure bank and print its sha256. Run before the reproduction."""

from __future__ import annotations

import sys

import polars as pl

from eci.config import REPO_ROOT
from eci.data.manifest import sha256_file
from eci.eval.dispatch import declared_inputs
from eci.simulate.declared_bank import build_declared_bank


def main() -> int:
    source = REPO_ROOT / "data" / "interim" / "sessions.parquet"
    if not source.exists():
        print(f"missing {source}", file=sys.stderr)
        return 1
    frame = pl.read_parquet(source)
    declared = declared_inputs(source)
    bank = build_declared_bank(frame, declared)
    dest = REPO_ROOT / "scenarios" / "bank_declared_K50.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    bank.write_parquet(dest)
    print(f"rows={bank.height} scenarios={bank['scenario_id'].n_unique()} sha256={sha256_file(dest)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
