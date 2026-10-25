"""Day-block bootstrap. A drawn day moves every session from that day together."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

LA = ZoneInfo("America/Los_Angeles")


def _local_day(moment: datetime) -> date:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(LA).date()


def _shift_local(moment: datetime, target: date) -> datetime:
    local = moment.astimezone(LA) if moment.tzinfo else moment.replace(tzinfo=timezone.utc).astimezone(LA)
    try:
        moved = local.replace(year=target.year, month=target.month, day=target.day)
    except ValueError:
        moved = (local + timedelta(hours=1)).replace(year=target.year, month=target.month, day=target.day)
    return moved.astimezone(timezone.utc)


def _shift_pair(connection: datetime, disconnect: datetime, target: date) -> tuple[datetime, datetime]:
    """Move a session onto ``target`` and keep the local dwell, including overnight stays."""
    if connection.tzinfo is None:
        connection = connection.replace(tzinfo=timezone.utc)
    if disconnect.tzinfo is None:
        disconnect = disconnect.replace(tzinfo=timezone.utc)
    extra_days = (_local_day(disconnect) - _local_day(connection)).days
    return _shift_local(connection, target), _shift_local(disconnect, target + timedelta(days=extra_days))


def day_block_bootstrap(
    sessions: pl.DataFrame,
    *,
    regime_start: date,
    regime_end: date,
    n_years: int,
    seed: int,
    skeleton_year: int = 2019,
) -> pl.DataFrame:
    """Resample whole local days, stratified by weekday and quarter.

    ``regime_end`` is exclusive. Each synthetic year uses the calendar of
    ``skeleton_year`` and draws a historical day from the matching stratum.
    """
    frame = sessions.filter(pl.col("kWhDelivered") > 0).with_columns(
        pl.col("connectionTime_utc").dt.convert_time_zone("America/Los_Angeles").dt.date().alias("local_date")
    )
    frame = frame.filter((pl.col("local_date") >= regime_start) & (pl.col("local_date") < regime_end))
    grouped: dict[date, list[dict]] = {}
    for row in frame.iter_rows(named=True):
        grouped.setdefault(row["local_date"], []).append(row)
    day = regime_start
    while day < regime_end:
        grouped.setdefault(day, [])
        day += timedelta(days=1)
    strata: dict[tuple[int, int], list[date]] = {}
    by_weekday: dict[int, list[date]] = {}
    for day in grouped:
        key = (day.weekday(), (day.month - 1) // 3 + 1)
        strata.setdefault(key, []).append(day)
        by_weekday.setdefault(day.weekday(), []).append(day)
    if not grouped:
        raise ValueError("regime window has no sessions")
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    skeleton = date(skeleton_year, 1, 1)
    for scenario_id in range(n_years):
        draw = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        for offset in range(366):
            target = skeleton + timedelta(days=offset)
            if target.year != skeleton_year:
                break
            key = (target.weekday(), (target.month - 1) // 3 + 1)
            pool = strata.get(key) or by_weekday.get(target.weekday()) or list(grouped)
            picked = pool[int(draw.integers(0, len(pool)))]
            for source in grouped[picked]:
                connection, disconnect = _shift_pair(source["connectionTime_utc"], source["disconnectTime_utc"], target)
                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "seed": seed,
                        "source_date": picked.isoformat(),
                        "sessionID": f"{scenario_id}-{source['sessionID']}-{target.isoformat()}",
                        "stationID": source.get("stationID"),
                        "connectionTime_utc": connection,
                        "disconnectTime_utc": disconnect,
                        "kWhDelivered": float(source["kWhDelivered"]),
                        "pilot_amp": source.get("pilot_amp"),
                    }
                )
    return pl.DataFrame(rows)
