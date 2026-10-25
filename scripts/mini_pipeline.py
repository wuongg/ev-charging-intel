"""Nightly mini-pipeline: 200 synthetic sessions, audit gate, one naive number."""

from __future__ import annotations

import sys

import polars as pl

from eci.data.audit import audit_gate
from eci.data.clean import clean_sessions
from eci.data.features import build_hourly_demand
from eci.data.parse import parse_sessions
from eci.data.sample import make_sessions
from eci.forecast.naive import seasonal_naive


def main() -> int:
    records = make_sessions(200, seed=7)
    parsed = parse_sessions(records)
    cleaned, funnel = clean_sessions(parsed)
    report = audit_gate(cleaned, funnel)
    if not report.passed:
        print("mini-pipeline audit gate failed")
        for reason in report.reasons:
            print(reason)
        return 1
    hourly = build_hourly_demand(cleaned, "caltech")
    forecast = seasonal_naive(hourly)
    number = forecast.select(pl.col("q50").drop_nulls().mean()).item()
    if number is None:
        number = forecast.select(pl.col("yhat").drop_nulls().mean()).item()
    print(f"MINI_PIPELINE_NUMBER={float(number):.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
