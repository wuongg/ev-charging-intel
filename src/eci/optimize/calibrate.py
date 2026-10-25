"""T2C: actual dwell given the declared window. Fit never sees the scored session."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from eci.eval.dispatch import _normalize_session_id, _source_key, declared_inputs
from eci.optimize.impute import empirical_quantile, unit_interval
from eci.optimize.mpc import ScenarioSession

LA = ZoneInfo("America/Los_Angeles")
# Declared dwell buckets, in hours. The last edge is exclusive of nothing: 12h+ sits in the tail.
BUCKET_EDGES = (0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 36.0)
DEFAULT_BETA = 0.5
MIN_N = 20


def fold_of(session_id: str) -> int:
    """0 or 1 from the source id. The same id always lands in the same fold."""
    return 0 if unit_interval(_normalize_session_id(_source_key(session_id))) < 0.5 else 1


def declared_bucket(hours: float) -> int:
    if hours <= 0:
        return 0
    for index, edge in enumerate(BUCKET_EDGES[1:]):
        if hours < edge:
            return index
    return len(BUCKET_EDGES) - 2


@dataclass
class HazardFold:
    """Empirical actual-dwell CDFs. Thin cells fall back to the hour, then the fold."""

    by_cell: dict[tuple[int, int], np.ndarray]
    by_hour: dict[int, np.ndarray]
    global_hours: np.ndarray
    min_n: int = MIN_N

    def pool(self, hour: int, bucket: int | None) -> np.ndarray:
        if bucket is not None:
            cell = self.by_cell.get((hour, bucket))
            if cell is not None and cell.size >= self.min_n:
                return cell
        hourly = self.by_hour.get(hour)
        if hourly is not None and hourly.size >= self.min_n:
            return hourly
        return self.global_hours

    def actual_hours(self, hour: int, declared_hours: float | None, beta: float) -> float:
        bucket = None if declared_hours is None else declared_bucket(declared_hours)
        return empirical_quantile(self.pool(hour, bucket), beta)


@dataclass
class HazardCorrector:
    """Two folds. A session is scored with the fold that did not train on it."""

    folds: tuple[HazardFold, HazardFold]
    beta: float = DEFAULT_BETA

    def hours_for(self, session: ScenarioSession) -> float:
        local = session.connection.astimezone(LA)
        declared = None
        if session.requested_departure is not None:
            declared = (session.requested_departure - session.connection).total_seconds() / 3600.0
            if declared <= 0:
                declared = None
        return self.folds[1 - fold_of(session.session_id)].actual_hours(local.hour, declared, self.beta)

    def apply(self, sessions: list[ScenarioSession], beta: float | None = None) -> list[ScenarioSession]:
        """Overwrite the believed deadline. Raw declared is never sent to the planner."""
        beta = self.beta if beta is None else float(beta)
        if not 0.0 < beta < 1.0:
            raise ValueError(f"beta must be in (0, 1), got {beta}")
        previous = self.beta
        self.beta = beta
        out: list[ScenarioSession] = []
        for session in sessions:
            hours = max(self.hours_for(session), 1e-6)
            out.append(replace(session, requested_departure=session.connection + timedelta(hours=hours)))
        self.beta = previous
        return out


def _fold_from_rows(rows: list[tuple[int, int, float]]) -> HazardFold:
    cells: dict[tuple[int, int], list[float]] = {}
    hours: dict[int, list[float]] = {}
    global_hours: list[float] = []
    for hour, bucket, actual in rows:
        if actual <= 0:
            continue
        cells.setdefault((hour, bucket), []).append(actual)
        hours.setdefault(hour, []).append(actual)
        global_hours.append(actual)
    if not global_hours:
        raise ValueError("hazard fold is empty")
    return HazardFold(
        by_cell={key: np.sort(np.asarray(values)) for key, values in cells.items()},
        by_hour={key: np.sort(np.asarray(values)) for key, values in hours.items()},
        global_hours=np.sort(np.asarray(global_hours)),
    )


def records_from_sessions(sessions: list[ScenarioSession]) -> list[tuple[str, int, int, float, float]]:
    """(id, hour, bucket, declared_h, actual_h) for sessions that declared a window."""
    rows = []
    for session in sessions:
        if session.requested_departure is None:
            continue
        declared = (session.requested_departure - session.connection).total_seconds() / 3600.0
        actual = (session.disconnect - session.connection).total_seconds() / 3600.0
        if declared <= 0 or actual <= 0:
            continue
        hour = session.connection.astimezone(LA).hour
        rows.append((session.session_id, hour, declared_bucket(declared), declared, actual))
    return rows


def corrector_from_records(
    rows: list[tuple[str, int, int, float, float]],
    beta: float = DEFAULT_BETA,
) -> HazardCorrector:
    """Fit one fold per half of the ids. Each id is held out of its scoring fold."""
    split: list[list[tuple[int, int, float]]] = [[], []]
    for session_id, hour, bucket, _declared, actual in rows:
        split[fold_of(session_id)].append((hour, bucket, actual))
    if not split[0] or not split[1]:
        raise ValueError("both folds need declared sessions")
    return HazardCorrector(folds=(_fold_from_rows(split[0]), _fold_from_rows(split[1])), beta=beta)


def corrector_from_parquet(path, beta: float = DEFAULT_BETA) -> HazardCorrector:
    """Caltech 2018 sessions that declared a departure. kWh == 0 is dropped."""
    from eci.simulate.declared_bank import declared_session_ids

    frame = pl.read_parquet(path)
    declared = declared_inputs(path)
    ids = set(declared_session_ids(frame, declared))
    subset = frame.filter(pl.col("sessionID").is_in(list(ids)))
    sessions: list[ScenarioSession] = []
    for row in subset.iter_rows(named=True):
        connection = row["connectionTime_utc"]
        disconnect = row["disconnectTime_utc"]
        if connection is None or disconnect is None:
            continue
        delta, _energy = declared.get(_normalize_session_id(row["sessionID"]), (None, None))
        if delta is None:
            continue
        sessions.append(
            ScenarioSession(
                session_id=str(row["sessionID"]),
                connection=connection,
                disconnect=disconnect,
                kwh=float(row["kWhDelivered"] or 0.0),
                p_max_kw=6.7,
                requested_departure=connection + delta,
                station_id=str(row.get("stationID") or ""),
            )
        )
    return corrector_from_records(records_from_sessions(sessions), beta=beta)
