"""Cleaning funnel R1–V6. Exclusions leave the frame; flags stay on the row."""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import polars as pl

from eci.constants import (
    F_CLOCK_SKEW,
    F_DST_CROSS,
    F_KWH_OOR,
    F_LONG_DWELL,
    F_TYPE_COERCED,
    F_UI_ABSURD,
    KWH_MAX,
    KWH_MIN,
    LONG_DWELL_HOURS,
)
from eci.data.parse import dumps_json, loads_inputs, select_snapshot

_OUTPUT_COLUMNS = [
    "_id",
    "clusterID",
    "connectionTime",
    "connectionTime_utc",
    "connectionTime_local",
    "disconnectTime",
    "disconnectTime_utc",
    "disconnectTime_local",
    "doneChargingTime",
    "doneChargingTime_utc",
    "doneChargingTime_local",
    "kWhDelivered",
    "sessionID",
    "siteID",
    "spaceID",
    "stationID",
    "timezone",
    "userID",
    "userInputs",
    "userInputs_last",
    "n_user_inputs",
    "extra_json",
    F_CLOCK_SKEW,
    F_LONG_DWELL,
    F_UI_ABSURD,
    F_DST_CROSS,
    F_TYPE_COERCED,
    F_KWH_OOR,
]


def _number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def input_is_absurd(event: dict) -> bool:
    """kWhRequested <= 0, minutesAvailable <= 5, or WhPerMile outside (0, 500]."""
    if "kWhRequested" in event:
        kwh = _number(event.get("kWhRequested"))
        if kwh is None or kwh <= 0:
            return True
    if "minutesAvailable" in event:
        minutes = _number(event.get("minutesAvailable"))
        if minutes is None or minutes <= 5:
            return True
    if "WhPerMile" in event:
        wh = _number(event.get("WhPerMile"))
        if wh is None or wh <= 0 or wh > 500:
            return True
    return False


def _as_utc(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return None
        return value.astimezone(timezone.utc)
    return None


def crosses_dst(connection: datetime | None, disconnect: datetime | None, timezone_name: str) -> bool:
    """True when UTC elapsed time differs from the local wall-clock elapsed time."""
    if connection is None or disconnect is None:
        return False
    zone = ZoneInfo(timezone_name)
    local_connection = connection.astimezone(zone).replace(tzinfo=None)
    local_disconnect = disconnect.astimezone(zone).replace(tzinfo=None)
    return (disconnect - connection) != (local_disconnect - local_connection)


def _funnel_row(stage: str, rule: str, action: str, n_in: int, n_excluded: int, n_flagged: int, n_out: int) -> dict:
    return {
        "stage": stage,
        "rule": rule,
        "action": action,
        "n_in": n_in,
        "n_excluded": n_excluded,
        "n_flagged": n_flagged,
        "n_out": n_out,
    }


def clean_sessions(
    df: pl.DataFrame,
    timezone_name: str = "America/Los_Angeles",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Apply R1–V6. Returns the kept frame and a funnel table.

    Timestamp columns are filtered from the input frame so the UTC and local
    dtypes survive. Flags and the decision-time snapshot are joined back on.
    """
    indexed = df.with_row_index("_eci_row")
    rows = indexed.to_dicts()
    funnel: list[dict] = []

    n_in = len(rows)
    parsed: list[dict] = []
    for row in rows:
        connection_missing = row.get("connectionTime_utc") is None or row.get("connection_bad")
        disconnect_missing = row.get("disconnectTime_utc") is None or row.get("disconnect_bad")
        done_raw = row.get("doneChargingTime")
        done_bad = bool(row.get("done_bad")) and done_raw not in (None, "")
        if connection_missing or disconnect_missing or done_bad:
            continue
        parsed.append(row)
    funnel.append(_funnel_row("R1", "parse_fail", "EXCLUDE", n_in, n_in - len(parsed), 0, len(parsed)))

    n_coerced = sum(1 for row in parsed if row.get("type_coerced"))
    for row in parsed:
        row[F_TYPE_COERCED] = bool(row.get("type_coerced"))
    funnel.append(_funnel_row("R2", "schema_coerce", "FLAG", len(parsed), 0, n_coerced, len(parsed)))

    seen: set[tuple] = set()
    deduped: list[dict] = []
    n_dup = 0
    for row in parsed:
        connection = _as_utc(row.get("connectionTime_utc"))
        key = (row.get("stationID"), connection.isoformat() if connection else None, row.get("spaceID"))
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        deduped.append(row)
    funnel.append(
        _funnel_row("R3", "dedupe_station_connection_space", "EXCLUDE", len(parsed), n_dup, 0, len(deduped))
    )

    n_dst = 0
    n_skew = 0
    for row in deduped:
        connection = _as_utc(row.get("connectionTime_utc"))
        disconnect = _as_utc(row.get("disconnectTime_utc"))
        done = _as_utc(row.get("doneChargingTime_utc"))
        row[F_DST_CROSS] = crosses_dst(connection, disconnect, timezone_name)
        row[F_CLOCK_SKEW] = bool(done is not None and disconnect is not None and done > disconnect)
        n_dst += int(row[F_DST_CROSS])
        n_skew += int(row[F_CLOCK_SKEW])
    funnel.append(_funnel_row("V1", "dst_cross", "FLAG", len(deduped), 0, n_dst, len(deduped)))
    funnel.append(_funnel_row("V2", "clock_skew", "FLAG", len(deduped), 0, n_skew, len(deduped)))

    ordered: list[dict] = []
    n_order = 0
    for row in deduped:
        connection = _as_utc(row.get("connectionTime_utc"))
        disconnect = _as_utc(row.get("disconnectTime_utc"))
        if connection is not None and disconnect is not None and disconnect < connection:
            n_order += 1
            continue
        ordered.append(row)
    funnel.append(_funnel_row("V3", "time_order", "EXCLUDE", len(deduped), n_order, 0, len(ordered)))

    n_kwh = 0
    n_dwell = 0
    n_absurd = 0
    for row in ordered:
        kwh = row.get("kWhDelivered")
        oor = isinstance(kwh, (int, float)) and not isinstance(kwh, bool) and (kwh < KWH_MIN or kwh > KWH_MAX)
        row[F_KWH_OOR] = bool(oor)
        n_kwh += int(oor)
        connection = _as_utc(row.get("connectionTime_utc"))
        disconnect = _as_utc(row.get("disconnectTime_utc"))
        dwell_hours = 0.0
        if connection is not None and disconnect is not None:
            dwell_hours = (disconnect - connection).total_seconds() / 3600.0
        row[F_LONG_DWELL] = dwell_hours > LONG_DWELL_HOURS
        n_dwell += int(row[F_LONG_DWELL])

        original = loads_inputs(row.get("userInputs"))
        valid = [event for event in original if isinstance(event, dict) and not input_is_absurd(event)]
        absurd = [event for event in original if isinstance(event, dict) and input_is_absurd(event)]
        row[F_UI_ABSURD] = len(absurd) > 0
        n_absurd += int(row[F_UI_ABSURD])
        snapshot = select_snapshot(valid, connection)
        row["userInputs_last"] = dumps_json(snapshot)

    funnel.append(_funnel_row("V4", "kwh_range", "FLAG", len(ordered), 0, n_kwh, len(ordered)))
    funnel.append(_funnel_row("V5", "long_dwell", "FLAG", len(ordered), 0, n_dwell, len(ordered)))
    funnel.append(_funnel_row("V6", "userinputs_absurd", "FLAG", len(ordered), 0, n_absurd, len(ordered)))

    cleaned = _assemble(indexed, ordered)
    funnel_df = pl.DataFrame(funnel)
    return cleaned, funnel_df


def _assemble(indexed: pl.DataFrame, ordered: list[dict]) -> pl.DataFrame:
    helpers = [
        column
        for column in ("type_coerced", "connection_bad", "disconnect_bad", "done_bad", "userInputs_last")
        if column in indexed.columns
    ]
    base = indexed.drop(helpers) if helpers else indexed
    if not ordered:
        empty = base.head(0)
        return empty.with_columns(
            [
                pl.lit(None, dtype=pl.Boolean).alias(flag)
                for flag in (
                    F_CLOCK_SKEW,
                    F_LONG_DWELL,
                    F_UI_ABSURD,
                    F_DST_CROSS,
                    F_TYPE_COERCED,
                    F_KWH_OOR,
                )
            ]
            + [pl.lit(None, dtype=pl.Utf8).alias("userInputs_last")]
        ).drop("_eci_row")

    flags = pl.DataFrame(
        {
            "_eci_row": [row["_eci_row"] for row in ordered],
            "_order": list(range(len(ordered))),
            F_CLOCK_SKEW: [bool(row[F_CLOCK_SKEW]) for row in ordered],
            F_LONG_DWELL: [bool(row[F_LONG_DWELL]) for row in ordered],
            F_UI_ABSURD: [bool(row[F_UI_ABSURD]) for row in ordered],
            F_DST_CROSS: [bool(row[F_DST_CROSS]) for row in ordered],
            F_TYPE_COERCED: [bool(row[F_TYPE_COERCED]) for row in ordered],
            F_KWH_OOR: [bool(row[F_KWH_OOR]) for row in ordered],
            "userInputs_last": [row["userInputs_last"] for row in ordered],
        }
    )
    joined = flags.join(base, on="_eci_row", how="left").sort("_order")
    keep = [column for column in _OUTPUT_COLUMNS if column in joined.columns]
    return joined.select(keep)
