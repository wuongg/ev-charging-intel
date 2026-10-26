"""Parse the a.s.r. / Twente office-parking EV sessions into the repo schema.

License: CC BY-NC-SA 4.0 (4TU.ResearchData DOI 10.4121/80ef3824-3f5d-4e45-8794-3b8791efbd13).
Demo-only. Do not redistribute the files. data/ is gitignored.
"""

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from eci.data.parse import format_rfc1123_utc

AMSTERDAM = ZoneInfo("Europe/Amsterdam")
TWENTE_TZ = "Europe/Amsterdam"

# a.s.r. living lab, Utrecht. 25 Aug 2020 – 2 Oct 2024.
TWENTE_DOI = "10.4121/80ef3824-3f5d-4e45-8794-3b8791efbd13"
TWENTE_UUID = "80ef3824-3f5d-4e45-8794-3b8791efbd13"
TWENTE_LICENSE = "CC BY-NC-SA 4.0"
TWENTE_NOTE = (
    "Twente/a.s.r. office-parking sessions, CC BY-NC-SA 4.0. Drift-demo only. "
    "Not redistributed. DOI 10.4121/80ef3824-3f5d-4e45-8794-3b8791efbd13."
)

_START_ALIASES = (
    "start",
    "starttime",
    "start_time",
    "startdatetime",
    "connectiontime",
    "plugintime",
    "arrival",
    "transactionstart",
    "utcstart",
    "start_utc",
)
_END_ALIASES = (
    "end",
    "endtime",
    "end_time",
    "enddatetime",
    "disconnecttime",
    "unplugtime",
    "departure",
    "transactionend",
    "utcend",
    "end_utc",
)
_KWH_ALIASES = (
    "kwh",
    "kwhdelivered",
    "energy",
    "energy_kwh",
    "volume",
    "chargedenergy",
    "energywh",
)
_ID_ALIASES = ("sessionid", "transactionid", "id", "session_id", "transaction_id")
_STATION_ALIASES = ("stationid", "chargepoint", "chargerid", "evseid", "socketid", "station")
_USER_ALIASES = ("userid", "cardid", "rfid", "user", "anonymizeduserid")


def _norm(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _pick(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    mapped = {_norm(col): col for col in columns}
    for alias in aliases:
        if alias in mapped:
            return mapped[alias]
    return None


def _as_utc(series: pl.Series) -> pl.Series:
    dtype = series.dtype
    if dtype == pl.Datetime:
        tz = getattr(dtype, "time_zone", None)
        if tz:
            return series.dt.convert_time_zone("UTC")
        # Naive office-parking stamps are local Amsterdam wall time.
        return series.dt.replace_time_zone(TWENTE_TZ).dt.convert_time_zone("UTC")
    text = series.cast(pl.Utf8)
    parsed = text.str.to_datetime(time_zone=TWENTE_TZ, strict=False, exact=False)
    return parsed.dt.convert_time_zone("UTC")


def sessions_from_frame(raw: pl.DataFrame) -> pl.DataFrame:
    """Map whatever the 4TU workbook called the columns onto the ACN-shaped schema."""
    start_col = _pick(raw.columns, _START_ALIASES)
    end_col = _pick(raw.columns, _END_ALIASES)
    kwh_col = _pick(raw.columns, _KWH_ALIASES)
    if start_col is None or end_col is None or kwh_col is None:
        raise ValueError(f"Twente columns not recognized: {raw.columns}")
    id_col = _pick(raw.columns, _ID_ALIASES)
    station_col = _pick(raw.columns, _STATION_ALIASES)
    user_col = _pick(raw.columns, _USER_ALIASES)
    kwh = raw[kwh_col].cast(pl.Float64)
    name = kwh_col.lower()
    if "wh" in name and "kwh" not in name:
        kwh = kwh / 1000.0
    connection_utc = _as_utc(raw[start_col])
    disconnect_utc = _as_utc(raw[end_col])
    n = raw.height
    session_id = (
        raw[id_col].cast(pl.Utf8)
        if id_col is not None
        else pl.Series("sessionID", [f"twente-{i}" for i in range(n)])
    )
    station = raw[station_col].cast(pl.Utf8) if station_col is not None else pl.Series("stationID", ["twente"] * n)
    user = raw[user_col].cast(pl.Utf8) if user_col is not None else pl.Series("userID", [None] * n)
    rfc_conn = [format_rfc1123_utc(moment) if moment is not None else None for moment in connection_utc.to_list()]
    rfc_disc = [format_rfc1123_utc(moment) if moment is not None else None for moment in disconnect_utc.to_list()]
    frame = pl.DataFrame(
        {
            "_id": session_id,
            "clusterID": ["twente"] * n,
            "connectionTime": rfc_conn,
            "connectionTime_utc": connection_utc,
            "disconnectTime": rfc_disc,
            "disconnectTime_utc": disconnect_utc,
            "doneChargingTime": rfc_disc,
            "doneChargingTime_utc": disconnect_utc,
            "kWhDelivered": kwh,
            "sessionID": session_id,
            "siteID": ["twente"] * n,
            "spaceID": station,
            "stationID": station,
            "timezone": [TWENTE_TZ] * n,
            "userID": user,
            "userInputs": [None] * n,
        }
    )
    return (
        frame.filter(
            pl.col("connectionTime_utc").is_not_null()
            & pl.col("disconnectTime_utc").is_not_null()
            & (pl.col("disconnectTime_utc") > pl.col("connectionTime_utc"))
            & pl.col("kWhDelivered").is_not_null()
        )
        .with_columns(
            pl.col("connectionTime_utc").dt.convert_time_zone(TWENTE_TZ).alias("connectionTime_local"),
            pl.col("disconnectTime_utc").dt.convert_time_zone(TWENTE_TZ).alias("disconnectTime_local"),
            pl.col("doneChargingTime_utc").dt.convert_time_zone(TWENTE_TZ).alias("doneChargingTime_local"),
        )
        .sort("connectionTime_utc")
    )


def load_twente_table(path: Path) -> pl.DataFrame:
    """Read csv/xlsx/parquet from a 4TU download. First sheet if a workbook."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pl.read_csv(path, infer_schema_length=5000, ignore_errors=True)
    if suffix in {".parquet"}:
        return pl.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        try:
            return pl.read_excel(path)
        except Exception:  # noqa: BLE001 — optional engine; demo falls back to ACN
            import pandas as pd

            return pl.from_pandas(pd.read_excel(path))
    raise ValueError(f"unsupported Twente file: {path}")


def parse_twente_path(path: Path) -> pl.DataFrame:
    return sessions_from_frame(load_twente_table(path))


def find_twente_file(raw_dir: Path) -> Path | None:
    """First tabular file under data/raw/twente, if the NC snapshot was downloaded."""
    root = Path(raw_dir) / "twente"
    if not root.exists():
        return None
    for pattern in ("*.csv", "*.xlsx", "*.xls", "*.parquet"):
        hits = sorted(root.rglob(pattern))
        if hits:
            return hits[0]
    return None


def hourly_demand(sessions: pl.DataFrame) -> pl.DataFrame:
    """Connection-hour kWh, Amsterdam wall clock, same columns as the ACN hourly frame."""
    tz = sessions.schema["connectionTime_local"].time_zone or TWENTE_TZ
    stamped = sessions.with_columns(pl.col("connectionTime_local").dt.truncate("1h").alias("ts_local_hour"))
    aggregated = (
        stamped.group_by("ts_local_hour")
        .agg(
            pl.col("kWhDelivered").sum().alias("kWh_delivered"),
            pl.len().alias("n_sessions"),
        )
        .sort("ts_local_hour")
    )
    if aggregated.height == 0:
        return aggregated.with_columns(pl.lit("twente").alias("site"))
    start = aggregated["ts_local_hour"].min()
    end = aggregated["ts_local_hour"].max()
    grid = pl.DataFrame({"ts_local_hour": pl.datetime_range(start, end, "1h", time_zone=tz, eager=True)})
    return (
        grid.join(aggregated, on="ts_local_hour", how="left")
        .with_columns(
            pl.col("kWh_delivered").fill_null(0.0),
            pl.col("n_sessions").fill_null(0).cast(pl.Int64),
            pl.lit("twente").alias("site"),
        )
        .select(["ts_local_hour", "site", "kWh_delivered", "n_sessions"])
        .sort("ts_local_hour")
    )


