"""Hourly demand attributed by connection time, with right-censoring."""

from __future__ import annotations

from datetime import datetime

import polars as pl

# ACN site ids used when the caller passes a site name.
SITE_ID_BY_NAME = {"caltech": "0002", "jpl": "0001"}


def build_hourly_demand(
    sessions: pl.DataFrame,
    site: str,
    window_end: datetime | None = None,
) -> pl.DataFrame:
    """Sum kWh into the connection hour.

    doneChargingTime is not an attribution key: using it leaks completion.
    A session that has not disconnected by ``window_end`` is dropped from the
    aggregate. The default window end is the latest disconnect in the filtered
    frame, so open sessions (null disconnect) are the ones removed.
    Missing hours inside the spanned range are explicit zeros.
    """
    ids = {site, SITE_ID_BY_NAME.get(site.lower(), site)}
    filtered = sessions.filter(pl.col("siteID").is_in(list(ids)))
    if window_end is None:
        window_end = filtered.select(pl.col("disconnectTime_utc").max()).item()
    if window_end is not None:
        filtered = filtered.filter(
            pl.col("disconnectTime_utc").is_not_null() & (pl.col("disconnectTime_utc") <= window_end)
        )
    else:
        filtered = filtered.filter(pl.col("disconnectTime_utc").is_not_null())

    empty = pl.DataFrame(
        schema={
            "ts_local_hour": pl.Datetime("us", "America/Los_Angeles"),
            "site": pl.Utf8,
            "kWh_delivered": pl.Float64,
            "n_sessions": pl.Int64,
        }
    )
    if filtered.height == 0:
        return empty

    tz = filtered.schema["connectionTime_local"].time_zone or "America/Los_Angeles"
    stamped = filtered.with_columns(
        pl.col("connectionTime_local").dt.truncate("1h").alias("ts_local_hour")
    )
    aggregated = (
        stamped.group_by("ts_local_hour")
        .agg(
            pl.col("kWhDelivered").sum().alias("kWh_delivered"),
            pl.len().alias("n_sessions"),
        )
        .sort("ts_local_hour")
    )
    start = aggregated.select(pl.col("ts_local_hour").min()).item()
    end = aggregated.select(pl.col("ts_local_hour").max()).item()
    grid = pl.DataFrame(
        {"ts_local_hour": pl.datetime_range(start, end, "1h", time_zone=tz, eager=True)}
    )
    return (
        grid.join(aggregated, on="ts_local_hour", how="left")
        .with_columns(
            pl.col("kWh_delivered").fill_null(0.0),
            pl.col("n_sessions").fill_null(0).cast(pl.Int64),
            pl.lit(site).alias("site"),
        )
        .select(["ts_local_hour", "site", "kWh_delivered", "n_sessions"])
        .sort("ts_local_hour")
    )
