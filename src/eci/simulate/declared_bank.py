"""A K=50 bank drawn only from Caltech sessions that declared a departure."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from eci.eval.dispatch import _normalize_session_id
from eci.simulate.bootstrap import day_block_bootstrap


def declared_session_ids(frame: pl.DataFrame, declared: dict) -> list[str]:
    """Caltech rows with kWh > 0 and a usable requestedDeparture delta."""
    kept: list[str] = []
    for row in frame.select(["sessionID", "siteID", "kWhDelivered"]).iter_rows(named=True):
        if row["siteID"] != "0002":
            continue
        if row["kWhDelivered"] is None or float(row["kWhDelivered"]) <= 0:
            continue
        delta, _energy = declared.get(_normalize_session_id(row["sessionID"]), (None, None))
        if delta is None:
            continue
        kept.append(row["sessionID"])
    return kept


def build_declared_bank(frame: pl.DataFrame, declared: dict, *, n_years: int = 50, seed: int = 7) -> pl.DataFrame:
    """Day-block bootstrap on the declared subset. Skeleton year is 2018."""
    ids = declared_session_ids(frame, declared)
    if not ids:
        raise ValueError("no declared-departure sessions")
    subset = frame.filter(pl.col("sessionID").is_in(ids))
    local = subset.select(
        pl.col("connectionTime_utc").dt.convert_time_zone("America/Los_Angeles").dt.date().alias("local_date")
    )
    start = local["local_date"].min()
    end = local["local_date"].max() + timedelta(days=1)
    if not isinstance(start, date) or not isinstance(end, date):
        raise ValueError("declared subset has no local dates")
    return day_block_bootstrap(
        subset,
        regime_start=start,
        regime_end=end,
        n_years=n_years,
        seed=seed,
        skeleton_year=2018,
    )
