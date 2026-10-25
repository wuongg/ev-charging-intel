"""T2v2 deadline: one deterministic draw from the station-hour dwell CDF."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from datetime import timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from eci.optimize.mpc import ScenarioSession

LA = ZoneInfo("America/Los_Angeles")
MIN_STRATUM = 30


def unit_interval(session_id: str) -> float:
    """A number in (0, 1) that depends only on the session id."""
    digest = hashlib.sha256(session_id.encode("utf-8")).digest()
    raw = int.from_bytes(digest[:8], "big")
    return (raw + 0.5) / 2**64


def empirical_quantile(sorted_hours: np.ndarray, unit: float) -> float:
    """Linear quantile of a sorted sample. ``unit`` is in (0, 1)."""
    if sorted_hours.size == 0:
        raise ValueError("dwell sample is empty")
    if sorted_hours.size == 1:
        return float(sorted_hours[0])
    position = float(unit) * (sorted_hours.size - 1)
    low = int(np.floor(position))
    high = min(low + 1, sorted_hours.size - 1)
    weight = position - low
    return float(sorted_hours[low] * (1.0 - weight) + sorted_hours[high] * weight)


@dataclass
class DwellSampler:
    """Empirical CDFs. A thin station-hour falls back to the hour, then the year."""

    strata: dict[tuple[str, int], np.ndarray]
    by_hour: dict[int, np.ndarray]
    global_hours: np.ndarray
    min_n: int = MIN_STRATUM

    def pool(self, station: str, hour: int) -> np.ndarray:
        stratum = self.strata.get((station, hour))
        if stratum is not None and stratum.size >= self.min_n:
            return stratum
        hourly = self.by_hour.get(hour)
        if hourly is not None and hourly.size >= self.min_n:
            return hourly
        return self.global_hours

    def sample_hours(self, session_id: str, station: str, hour: int) -> float:
        return empirical_quantile(self.pool(station, hour), unit_interval(session_id))

    def quantile_hours(self, session: ScenarioSession, tau: float) -> float:
        """Station-hour dwell quantile. The same tau always returns the same hours."""
        if not 0.0 < tau < 1.0:
            raise ValueError(f"tau must be in (0, 1), got {tau}")
        local = session.connection.astimezone(LA)
        return empirical_quantile(self.pool(session.station_id, local.hour), tau)

    def apply_quantile(self, sessions: list[ScenarioSession], tau: float) -> list[ScenarioSession]:
        """Replace a missing deadline with the stratum quantile. A declared one stays."""
        out: list[ScenarioSession] = []
        for session in sessions:
            if session.requested_departure is not None:
                out.append(session)
                continue
            hours = self.quantile_hours(session, tau)
            out.append(replace(session, requested_departure=session.connection + timedelta(hours=hours)))
        return out

    def apply(self, sessions: list[ScenarioSession]) -> list[ScenarioSession]:
        """Fill a missing requested departure. A declared one is left as it is."""
        out: list[ScenarioSession] = []
        for session in sessions:
            if session.requested_departure is not None:
                out.append(session)
                continue
            local = session.connection.astimezone(LA)
            hours = self.sample_hours(session.session_id, session.station_id, local.hour)
            out.append(replace(session, requested_departure=session.connection + timedelta(hours=hours)))
        return out


def sampler_from_frame(frame: pl.DataFrame) -> DwellSampler:
    """``frame`` has station_id and dwell_h columns, already filtered."""
    strata: dict[tuple[str, int], list[float]] = {}
    by_hour: dict[int, list[float]] = {}
    global_hours: list[float] = []
    for station, hour, dwell in frame.select(["station_id", "hour", "dwell_h"]).iter_rows():
        hours = float(dwell)
        if hours <= 0:
            continue
        strata.setdefault((str(station), int(hour)), []).append(hours)
        by_hour.setdefault(int(hour), []).append(hours)
        global_hours.append(hours)
    if not global_hours:
        raise ValueError("dwell frame is empty")
    return DwellSampler(
        strata={key: np.sort(np.asarray(values)) for key, values in strata.items()},
        by_hour={key: np.sort(np.asarray(values)) for key, values in by_hour.items()},
        global_hours=np.sort(np.asarray(global_hours)),
    )


def sampler_from_parquet(path, *, site: str = "0002", year_before: int = 2019) -> DwellSampler:
    """Caltech dwells from the year before the frozen bank. kWh == 0 is excluded."""
    frame = pl.read_parquet(path).filter(
        (pl.col("siteID") == site)
        & (pl.col("kWhDelivered") > 0)
        & (pl.col("connectionTime_utc").dt.year() < year_before)
    )
    local = pl.col("connectionTime_utc").dt.convert_time_zone("America/Los_Angeles")
    dwell = (pl.col("disconnectTime_utc") - pl.col("connectionTime_utc")).dt.total_seconds() / 3600.0
    prepared = frame.select(
        pl.col("stationID").alias("station_id"),
        local.dt.hour().alias("hour"),
        dwell.alias("dwell_h"),
    ).filter(pl.col("dwell_h") > 0)
    return sampler_from_frame(prepared)
