"""Build the K=50 day-block bank and record its hash before any policy run."""

from __future__ import annotations

import sys
from datetime import date

import polars as pl
import yaml

from eci.config import REPO_ROOT
from eci.data.manifest import write_manifest_entry
from eci.simulate.scenario_bank import make_scenario_bank


def _year_count(sessions: pl.DataFrame, year: int) -> int:
    local = sessions.with_columns(
        pl.col("connectionTime_utc").dt.convert_time_zone("America/Los_Angeles").dt.year().alias("year")
    )
    return local.filter((pl.col("year") == year) & (pl.col("kWhDelivered") > 0) & (pl.col("siteID") == "0002")).height


def main() -> int:
    path = REPO_ROOT / "data" / "interim" / "sessions.parquet"
    if not path.exists():
        print(f"missing {path}", file=sys.stderr)
        return 1
    sessions = pl.read_parquet(path)
    caltech = sessions.filter(pl.col("siteID") == "0002")
    regime_start = date(2019, 1, 1)
    regime_end = date(2020, 1, 1)
    historical = _year_count(caltech, 2019)
    if historical < 1000:
        print(f"2019 caltech sessions={historical}; regime is not a full year", file=sys.stderr)
        return 1
    dest = REPO_ROOT / "scenarios" / "bank_K50.parquet"
    summary = make_scenario_bank(
        caltech,
        dest,
        k=50,
        seed=7,
        regime_start=regime_start,
        regime_end=regime_end,
    )
    relative = abs(summary["mean_sessions_per_year"] - historical) / historical
    print(f"historical_2019_sessions={historical}")
    print(f"mean_sessions_per_year={summary['mean_sessions_per_year']:.2f}")
    print(f"relative_gap={relative:.4%}")
    print(f"bank_hash={summary['sha256']}")
    print(f"n_rows={summary['n_rows']}")
    if relative >= 0.05:
        print("bank mean is more than 5% from the 2019 count", file=sys.stderr)
        return 1
    from datetime import datetime, timezone

    write_manifest_entry(
        {
            "file": "scenarios/bank_K50.parquet",
            "sha256": summary["sha256"],
            "source_url": "day-block bootstrap of data/interim/sessions.parquet",
            "downloaded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "rows_recovered": summary["n_rows"],
            "truncated": False,
        }
    )
    protocol_path = REPO_ROOT / "configs" / "protocol.yaml"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8")) or {}
    protocol["bank_hash"] = summary["sha256"]
    protocol["regime"] = {
        "start": "2019-01-01",
        "end": "2020-01-01",
        "amendment": (
            "Day-2 Kaggle mirror ended 2018-11-13, so the pre-registered 2019-06 "
            "window had 0 origins. Path B1 extends the span through 2020. The bank "
            "uses calendar 2019, the first full pre-COVID year inside that span."
        ),
    }
    protocol["frozen"] = {"embargo_hours": 336, "penalty_unmet": 1000.0, "committed_k": 1}
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"wrote {dest}")
    print(f"wrote {protocol_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
