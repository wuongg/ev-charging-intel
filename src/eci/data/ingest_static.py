"""Parse the ACN-Data-Static per-session csv.gz snapshot (path B1).

Each file is one charging session. Minute profiles are streamed to parquet
one garage at a time so the working set stays a single session plus a small
row-group buffer. ``iced_fraction`` is the time-weighted share of samples
whose charging current is below 1 A. That is the minute-level ICED rate.
The day-2 session proxy stays beside it.
"""

from __future__ import annotations

import csv
import gzip
import logging
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

from eci.data.parse import format_rfc1123_utc, parse_rfc1123_utc

LOGGER = logging.getLogger(__name__)

CURRENT_ZERO_AMPS = 1.0
MAX_GAP_SECONDS = 300.0
SITE_ID_BY_NAME = {"caltech": "0002", "jpl": "0001"}
_FILENAME = re.compile(r"^(?P<station>.+)-(\d{4}-\d{2}-\d{2}T.+)\.csv\.gz$")

PROFILE_SCHEMA = pa.schema(
    [
        ("sessionID", pa.string()),
        ("stationID", pa.string()),
        ("site", pa.string()),
        ("garage", pa.string()),
        ("ts_utc", pa.timestamp("us", tz="UTC")),
        ("current_a", pa.float64()),
        ("pilot_a", pa.float64()),
        ("power_kw", pa.float64()),
        ("energy_kwh", pa.float64()),
    ]
)


def _parse_timestamp(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _station_from_name(name: str) -> str:
    match = _FILENAME.match(name)
    if match:
        return match.group("station")
    return name.removesuffix(".csv.gz")


def _as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_session_file(path: Path, site: str, garage: str) -> tuple[dict, list[dict]]:
    """One csv.gz. Returns a raw session record and 1-minute profile rows."""
    station = _station_from_name(path.name)
    session_id = path.name.removesuffix(".csv.gz")
    first: datetime | None = None
    last: datetime | None = None
    prev_ts: datetime | None = None
    prev_current: float | None = None
    iced_seconds = 0.0
    total_seconds = 0.0
    max_energy = 0.0
    max_pilot = 0.0
    buckets: dict[datetime, list[float]] = {}
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"empty {path}") from exc
        index = {name: position for position, name in enumerate(header)}
        ts_i = index[""] if "" in index else index.get("timestamp", 0)
        cur_i = index.get("Charging Current (A)")
        pilot_i = index.get("Actual Pilot (A)")
        power_i = index.get("Power (kW)")
        energy_i = index.get("Energy Delivered (kWh)")

        def cell(row: list[str], col: int | None) -> str | None:
            if col is None or col >= len(row):
                return None
            return row[col]

        for row in reader:
            stamp = cell(row, ts_i)
            if not stamp:
                continue
            try:
                moment = _parse_timestamp(stamp)
            except ValueError:
                continue
            current = _as_float(cell(row, cur_i)) or 0.0
            pilot = _as_float(cell(row, pilot_i)) or 0.0
            power = _as_float(cell(row, power_i)) or 0.0
            energy = _as_float(cell(row, energy_i)) or 0.0
            if first is None:
                first = moment
            last = moment
            max_energy = max(max_energy, energy)
            max_pilot = max(max_pilot, pilot)
            if prev_ts is not None and prev_current is not None:
                gap = (moment - prev_ts).total_seconds()
                if 0 < gap <= MAX_GAP_SECONDS:
                    total_seconds += gap
                    if abs(prev_current) < CURRENT_ZERO_AMPS:
                        iced_seconds += gap
            prev_ts = moment
            prev_current = current
            minute = moment.replace(second=0, microsecond=0)
            slot = buckets.get(minute)
            if slot is None:
                buckets[minute] = [current, pilot, power, energy, 1.0]
            else:
                slot[0] += current
                slot[1] = max(slot[1], pilot)
                slot[2] += power
                slot[3] = max(slot[3], energy)
                slot[4] += 1.0
    if first is None or last is None:
        raise ValueError(f"no timestamps in {path}")
    iced = (iced_seconds / total_seconds) if total_seconds else None
    profiles = [
        {
            "sessionID": session_id,
            "stationID": station,
            "site": site,
            "garage": garage,
            "ts_utc": minute,
            "current_a": slot[0] / slot[4],
            "pilot_a": slot[1],
            "power_kw": slot[2] / slot[4],
            "energy_kwh": slot[3],
        }
        for minute, slot in buckets.items()
    ]
    record = {
        "_id": None,
        "clusterID": None,
        "connectionTime": format_rfc1123_utc(first),
        "disconnectTime": format_rfc1123_utc(last),
        "doneChargingTime": None,
        "kWhDelivered": max_energy,
        "sessionID": session_id,
        "siteID": SITE_ID_BY_NAME.get(site, site),
        "spaceID": None,
        "stationID": station,
        "timezone": "America/Los_Angeles",
        "userID": None,
        "userInputs": None,
        "extra": {
            "source": "b1",
            "garage": garage,
            "iced_fraction": iced,
            "pilot_amp": max_pilot,
            "current_zero_amps": CURRENT_ZERO_AMPS,
        },
    }
    return record, profiles


def _parse_batch(task: tuple[str, str, list[str], str | None]) -> tuple[list[dict], int, int]:
    """Parse one batch in a worker. Minute rows go to a shard, not back to the parent."""
    site, garage, paths, shard = task
    records: list[dict] = []
    rows: list[dict] = []
    failed = 0
    for raw_path in paths:
        try:
            record, profiles = parse_session_file(Path(raw_path), site, garage)
        except (OSError, ValueError, gzip.BadGzipFile):
            failed += 1
            continue
        records.append(record)
        if shard is not None and profiles:
            rows.extend(profiles)
    if shard is not None and rows:
        dest = Path(shard)
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows, schema=PROFILE_SCHEMA), dest)
    return records, failed, len(paths)


def _combine_shards(profile_dir: Path, garage: str, shards: list[Path]) -> None:
    dest = profile_dir / f"{garage}.parquet"
    if dest.exists():
        dest.unlink()
    writer: pq.ParquetWriter | None = None
    try:
        for shard in shards:
            table = pq.read_table(shard)
            if writer is None:
                writer = pq.ParquetWriter(dest, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    for shard in shards:
        shard.unlink(missing_ok=True)


def parse_static_repo(root: Path, profile_dir: Path | None = None) -> list[dict]:
    """Walk garage directories and return raw session records.

    Files are parsed in a process pool. Minute rows are written to
    ``profile_dir/<garage>.parquet`` when that directory is given. Files
    that fail to parse are counted and skipped.
    """
    root = Path(root)
    series_root = root / "time series data"
    if not series_root.exists():
        series_root = root
    shard_root = None
    if profile_dir is not None:
        profile_dir = Path(profile_dir)
        profile_dir.mkdir(parents=True, exist_ok=True)
        for stale in profile_dir.glob("*.parquet"):
            stale.unlink()
        shard_root = profile_dir / "_shards"
        if shard_root.exists():
            shutil.rmtree(shard_root)
        shard_root.mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[str, str, list[str], str | None]] = []
    shards_by_garage: dict[str, list[Path]] = {}
    batch = 200
    for site_dir in sorted(path for path in series_root.iterdir() if path.is_dir()):
        for garage_dir in sorted(path for path in site_dir.iterdir() if path.is_dir()):
            files = [str(path) for path in garage_dir.iterdir() if path.name.endswith(".csv.gz")]
            garage = garage_dir.name
            for start in range(0, len(files), batch):
                chunk = files[start : start + batch]
                shard = None
                if shard_root is not None:
                    shard_path = shard_root / f"{site_dir.name}_{garage}_{start:06d}.parquet"
                    shard = str(shard_path)
                    shards_by_garage.setdefault(garage, []).append(shard_path)
                tasks.append((site_dir.name, garage, chunk, shard))
    records: list[dict] = []
    failed = 0
    seen = 0
    logged = 0
    workers = min(8, os.cpu_count() or 4)
    LOGGER.info("static tasks=%d workers=%d", len(tasks), workers)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_parse_batch, task) for task in tasks]
        for future in as_completed(futures):
            batch_records, batch_failed, batch_seen = future.result()
            records.extend(batch_records)
            failed += batch_failed
            seen += batch_seen
            step = seen // 2000
            if step > logged:
                logged = step
                LOGGER.info("static files=%d sessions=%d failed=%d", seen, len(records), failed)
    if shard_root is not None:
        for garage, shards in shards_by_garage.items():
            present = [path for path in shards if path.exists()]
            if present:
                _combine_shards(profile_dir, garage, present)
        shutil.rmtree(shard_root, ignore_errors=True)
    LOGGER.info("static done files=%d sessions=%d failed=%d", seen, len(records), failed)
    return records


def _connection_minute(record: dict) -> datetime | None:
    raw = record.get("connectionTime")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        moment = parse_rfc1123_utc(raw)
    except ValueError:
        try:
            moment = _parse_timestamp(raw)
        except ValueError:
            return None
    return moment.replace(second=0, microsecond=0)


def merge_session_records(mirror_records: list[dict], static_records: list[dict]) -> tuple[list[dict], dict]:
    """Attach B1 ICED fields onto B2 rows when the session matches.

    Match sessionID first, then stationID plus connection minute. A B1-only
    session keeps ``userInputs`` null. Matched rows keep the mirror's
    userInputs, doneChargingTime, and sessionID.
    """
    by_id: dict[str, dict] = {}
    by_key: dict[tuple[str, datetime], dict] = {}
    for record in static_records:
        session_id = record.get("sessionID")
        if isinstance(session_id, str):
            by_id.setdefault(session_id, record)
        minute = _connection_minute(record)
        station = record.get("stationID")
        if minute is not None and isinstance(station, str):
            by_key.setdefault((station, minute), record)

    matched_id = 0
    matched_time = 0
    used: set[int] = set()
    merged: list[dict] = []
    for record in mirror_records:
        static = None
        session_id = record.get("sessionID")
        if isinstance(session_id, str) and session_id in by_id:
            static = by_id[session_id]
            matched_id += 1
        else:
            minute = _connection_minute(record)
            station = record.get("stationID")
            if minute is not None and isinstance(station, str):
                static = by_key.get((station, minute))
                if static is not None:
                    matched_time += 1
        if static is None:
            merged.append(record)
            continue
        used.add(id(static))
        combined = dict(record)
        extra = dict(static.get("extra") or {})
        extra["source"] = "b1+b2"
        extra["b1_sessionID"] = static.get("sessionID")
        combined["extra"] = extra
        merged.append(combined)
    b1_only = 0
    for record in static_records:
        if id(record) in used:
            continue
        b1_only += 1
        merged.append(record)
    stats = {
        "n_mirror": len(mirror_records),
        "n_static": len(static_records),
        "n_matched_session_id": matched_id,
        "n_matched_station_time": matched_time,
        "n_b1_only": b1_only,
        "n_merged": len(merged),
    }
    return merged, stats


def extra_measures(extra_json: str | None) -> tuple[float | None, float | None]:
    """Read iced_fraction and pilot_amp stored on a session extra blob."""
    if not extra_json:
        return None, None
    try:
        payload = orjson.loads(extra_json)
    except orjson.JSONDecodeError:
        return None, None
    if not isinstance(payload, dict):
        return None, None
    nested = payload.get("extra")
    if "iced_fraction" not in payload and isinstance(nested, dict):
        payload = nested
    iced = payload.get("iced_fraction")
    pilot = payload.get("pilot_amp")
    return (
        float(iced) if isinstance(iced, (int, float)) else None,
        float(pilot) if isinstance(pilot, (int, float)) else None,
    )


def attach_b1_columns(frame):
    """Add iced_fraction and pilot_amp parsed from extra_json. Import-light on purpose."""
    import polars as pl

    iced: list[float | None] = []
    pilot: list[float | None] = []
    for value in frame["extra_json"].to_list():
        iced_value, pilot_value = extra_measures(value)
        iced.append(iced_value)
        pilot.append(pilot_value)
    return frame.with_columns(
        pl.Series("iced_fraction", iced, dtype=pl.Float64),
        pl.Series("pilot_amp", pilot, dtype=pl.Float64),
    )


def coverage_note(garages: list[str]) -> str:
    """Human-readable scope. JPL missing means the fallback garage only."""
    names = ", ".join(garages) if garages else "none"
    return f"B1 garages: {names}"


def list_garages(root: Path) -> list[str]:
    series_root = Path(root) / "time series data"
    if not series_root.exists():
        return []
    found = []
    for site_dir in sorted(path for path in series_root.iterdir() if path.is_dir()):
        for garage_dir in sorted(path for path in site_dir.iterdir() if path.is_dir()):
            if any(garage_dir.glob("*.csv.gz")):
                found.append(f"{site_dir.name}/{garage_dir.name}")
    return found


def span_months(start: datetime, end: datetime) -> float:
    return (end - start).total_seconds() / (86400.0 * 30.437)
