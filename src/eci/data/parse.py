"""Parse ACN sessions into timezone-aware UTC and site-local timestamps.

RFC1123 ``GMT`` is mapped to UTC. ``strptime`` ``%Z`` is locale-dependent and is
not used. A naive timestamp is an error, never an assumption.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from zoneinfo import ZoneInfo

import orjson
import polars as pl

from eci.data.schema_contract import coerce_session

LOGGER = logging.getLogger(__name__)

_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTHS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)


def format_rfc1123_utc(moment: datetime) -> str:
    """English RFC1123, independent of the process locale."""
    utc = moment.astimezone(timezone.utc)
    return (
        f"{_WEEKDAYS[utc.weekday()]}, {utc.day:02d} {_MONTHS[utc.month - 1]} "
        f"{utc.year:04d} {utc.hour:02d}:{utc.minute:02d}:{utc.second:02d} GMT"
    )


def parse_rfc1123_utc(value: str) -> datetime:
    """Parse an RFC1123 instant. GMT/UT become UTC. Naive results are rejected."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("empty timestamp")
    parsed = parsedate_to_datetime(value.strip())
    if parsed is None:
        raise ValueError(f"unparseable timestamp: {value!r}")
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp missing timezone: {value!r}")
    return parsed.astimezone(timezone.utc)


def _parse_optional(value: object) -> tuple[datetime | None, bool]:
    """Return (utc_or_none, failed). Null and blank are absent, not failures."""
    if value is None:
        return None, False
    if isinstance(value, str) and not value.strip():
        return None, False
    try:
        return parse_rfc1123_utc(str(value)), False
    except (ValueError, TypeError, IndexError, OverflowError):
        return None, True


def input_event_time(event: dict) -> datetime | None:
    """``createdOn`` when present, otherwise ACN's ``modifiedAt``."""
    for key in ("createdOn", "modifiedAt"):
        raw = event.get(key)
        if not raw:
            continue
        try:
            return parse_rfc1123_utc(str(raw))
        except ValueError:
            continue
    return None


def select_snapshot(inputs: list | None, connection_utc: datetime | None) -> dict | None:
    """Decision-time userInputs entry.

    Prefer the latest event whose timestamp is still <= connectionTime.
    The Caltech mirror usually stamps ``modifiedAt`` a few seconds *after*
    plug-in and has no ``createdOn``. When every stamp is after connection,
    the earliest event is the arrival declaration; later edits are post-hoc
    and are not the snapshot.
    """
    if not inputs:
        return None
    stamped: list[tuple[datetime, int, dict]] = []
    unstamped: list[tuple[int, dict]] = []
    for index, event in enumerate(inputs):
        if not isinstance(event, dict):
            continue
        moment = input_event_time(event)
        if moment is None:
            unstamped.append((index, event))
        else:
            stamped.append((moment, index, event))

    if connection_utc is not None and stamped:
        eligible = [item for item in stamped if item[0] <= connection_utc]
        if eligible:
            eligible.sort(key=lambda item: (item[0], item[1]))
            return eligible[-1][2]
        stamped.sort(key=lambda item: (item[0], item[1]))
        return stamped[0][2]
    if unstamped:
        return unstamped[-1][1]
    if stamped:
        stamped.sort(key=lambda item: (item[0], item[1]))
        return stamped[-1][2]
    return None


def dumps_json(value: object) -> str | None:
    if value is None:
        return None
    return orjson.dumps(value).decode("utf-8")


def loads_inputs(value: object) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    if isinstance(value, str):
        if not value.strip():
            return []
        loaded = orjson.loads(value)
        if loaded is None:
            return []
        if isinstance(loaded, dict):
            return [loaded]
        if isinstance(loaded, list):
            return loaded
    return []


def _epoch_us(moment: datetime | None) -> int | None:
    if moment is None:
        return None
    utc = moment.astimezone(timezone.utc)
    return int(utc.timestamp()) * 1_000_000 + utc.microsecond


def parse_sessions(records: list[dict], timezone_name: str = "America/Los_Angeles") -> pl.DataFrame:
    """Coerce and parse. Unparseable connection/disconnect stay null for the R1 gate.

    Both UTC and site-local columns are kept. Local time is ``timezone_name``
    from the site config (default America/Los_Angeles), not a naive clock.
    """
    ZoneInfo(timezone_name)  # fail fast on a bad site config
    rows: list[dict] = []
    for raw in records:
        try:
            coerced, type_coerced = coerce_session(raw)
        except (TypeError, AssertionError) as exc:
            LOGGER.warning("schema reject: %s", exc)
            coerced = {key: None for key in (
                "_id",
                "clusterID",
                "connectionTime",
                "disconnectTime",
                "doneChargingTime",
                "sessionID",
                "siteID",
                "spaceID",
                "stationID",
                "timezone",
                "userID",
            )}
            coerced["kWhDelivered"] = None
            coerced["userInputs"] = None
            coerced["extra"] = None
            type_coerced = True

        connection, connection_bad = _parse_optional(coerced.get("connectionTime"))
        disconnect, disconnect_bad = _parse_optional(coerced.get("disconnectTime"))
        done, done_bad = _parse_optional(coerced.get("doneChargingTime"))
        inputs = coerced.get("userInputs")
        input_list = inputs if isinstance(inputs, list) else []
        snapshot = select_snapshot(input_list, connection)
        rows.append(
            {
                "_id": coerced.get("_id"),
                "clusterID": coerced.get("clusterID"),
                "connectionTime": coerced.get("connectionTime"),
                "disconnectTime": coerced.get("disconnectTime"),
                "doneChargingTime": coerced.get("doneChargingTime"),
                "kWhDelivered": coerced.get("kWhDelivered"),
                "sessionID": coerced.get("sessionID"),
                "siteID": coerced.get("siteID"),
                "spaceID": coerced.get("spaceID"),
                "stationID": coerced.get("stationID"),
                "timezone": coerced.get("timezone"),
                "userID": coerced.get("userID"),
                "userInputs": dumps_json(input_list if inputs is not None else None),
                "userInputs_last": dumps_json(snapshot),
                "n_user_inputs": len(input_list) if inputs is not None else 0,
                "extra_json": dumps_json(coerced.get("extra")),
                "type_coerced": bool(type_coerced),
                "connection_bad": connection_bad,
                "disconnect_bad": disconnect_bad,
                "done_bad": done_bad,
                "connectionTime_utc_us": _epoch_us(connection),
                "disconnectTime_utc_us": _epoch_us(disconnect),
                "doneChargingTime_utc_us": _epoch_us(done),
            }
        )

    schema = {
        "_id": pl.Utf8,
        "clusterID": pl.Utf8,
        "connectionTime": pl.Utf8,
        "disconnectTime": pl.Utf8,
        "doneChargingTime": pl.Utf8,
        "kWhDelivered": pl.Float64,
        "sessionID": pl.Utf8,
        "siteID": pl.Utf8,
        "spaceID": pl.Utf8,
        "stationID": pl.Utf8,
        "timezone": pl.Utf8,
        "userID": pl.Utf8,
        "userInputs": pl.Utf8,
        "userInputs_last": pl.Utf8,
        "n_user_inputs": pl.Int64,
        "extra_json": pl.Utf8,
        "type_coerced": pl.Boolean,
        "connection_bad": pl.Boolean,
        "disconnect_bad": pl.Boolean,
        "done_bad": pl.Boolean,
        "connectionTime_utc_us": pl.Int64,
        "disconnectTime_utc_us": pl.Int64,
        "doneChargingTime_utc_us": pl.Int64,
    }
    frame = pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)

    def _as_utc(column: str, alias: str) -> pl.Expr:
        return (
            pl.col(column).cast(pl.Datetime("us", "UTC"), strict=False).alias(alias)
        )

    return frame.with_columns(
        _as_utc("connectionTime_utc_us", "connectionTime_utc"),
        _as_utc("disconnectTime_utc_us", "disconnectTime_utc"),
        _as_utc("doneChargingTime_utc_us", "doneChargingTime_utc"),
    ).with_columns(
        pl.col("connectionTime_utc").dt.convert_time_zone(timezone_name).alias("connectionTime_local"),
        pl.col("disconnectTime_utc").dt.convert_time_zone(timezone_name).alias("disconnectTime_local"),
        pl.col("doneChargingTime_utc")
        .dt.convert_time_zone(timezone_name)
        .alias("doneChargingTime_local"),
    ).drop(
        "connectionTime_utc_us",
        "disconnectTime_utc_us",
        "doneChargingTime_utc_us",
    )
