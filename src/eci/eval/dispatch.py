"""One synthetic year: P0/P1/P2b at T2, P3 at T2, P4 at T3, plus the day's minimum cap."""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from eci.eval.metrics import demand_cost, merge_peaks, monthly_peaks, plug_binding_fraction
from eci.optimize.mpc import MPCLoop, ScenarioSession
from eci.optimize.policies import charge_on_arrival, cheapest_slots_greedy, earliest_deadline_first
from eci.optimize.solve import solve_epoch_fast
from eci.optimize.state import FleetState, Vehicle

CLOCK = "America/Los_Angeles"
LA = ZoneInfo(CLOCK)


def _fast(penalty: float):
    def policy(state, slots, prices, cap, dt):
        return solve_epoch_fast(state, slots, prices, cap, dt, penalty)

    return policy


def p_max_kw(pilot, nominal_volts: float, default_kw: float) -> float:
    if isinstance(pilot, (int, float)) and not isinstance(pilot, bool) and pilot > 1:
        return float(pilot) * nominal_volts / 1000.0
    return default_kw


def _source_key(session_id: str) -> str:
    """Bank ids are ``{scenario}-{source id}-{YYYY-MM-DD}``. Source ids use hyphens."""
    text = str(session_id)
    head, _, body = text.partition("-")
    if head.isdigit() and body:
        text = body
    if len(text) >= 11 and text[-11] == "-" and text[-10:-8].isdigit() and text[-5:-3].isdigit():
        text = text[:-11]
    return text


def _normalize_session_id(session_id: str) -> str:
    return (
        str(session_id)
        .replace("_", "-")
        .replace(" ", "T")
        .replace(":", "-")
        .replace(".", "-")
    )


def sessions_from_rows(
    rows: list[dict],
    nominal_volts: float,
    default_kw: float,
    declared: dict | None = None,
) -> list[ScenarioSession]:
    """kWh == 0 rows are not an energy need. The frozen bank already dropped them.

    ``declared`` maps sessionID to (requested_departure, kWh_requested). A missing
    deadline stays None so the MPC loop imputes connection + median dwell.
    """
    sessions: list[ScenarioSession] = []
    declared = declared or {}
    for row in rows:
        connection = row["connectionTime_utc"]
        disconnect = row["disconnectTime_utc"]
        kwh = float(row["kWhDelivered"])
        if disconnect <= connection or kwh <= 0:
            continue
        if connection.tzinfo is None:
            connection = connection.replace(tzinfo=ZoneInfo("UTC"))
        if disconnect.tzinfo is None:
            disconnect = disconnect.replace(tzinfo=ZoneInfo("UTC"))
        departure, _requested_kwh = declared.get(_source_key(str(row["sessionID"])), (None, None))
        if isinstance(departure, timedelta):
            departure = connection + departure
        if departure is not None and departure <= connection:
            departure = None
        sessions.append(
            ScenarioSession(
                session_id=str(row["sessionID"]),
                connection=connection,
                disconnect=disconnect,
                kwh=kwh,
                p_max_kw=p_max_kw(row.get("pilot_amp"), nominal_volts, default_kw),
                requested_departure=departure,
                kwh_requested=None,
                station_id=str(row.get("stationID") or ""),
            )
        )
    return sessions


def declared_inputs(path) -> dict[str, tuple[timedelta | None, float | None]]:
    """Last userInputs entry per session. Absent or absurd requests stay unset."""
    import json
    from email.utils import parsedate_to_datetime

    if not path.exists():
        return {}
    frame = pl.read_parquet(path).select(["sessionID", "userInputs", "connectionTime_utc"])
    found: dict[str, tuple[timedelta | None, float | None]] = {}
    for row in frame.iter_rows(named=True):
        raw = row["userInputs"]
        if raw is None or raw == "":
            continue
        try:
            payload = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list) or not payload:
            continue
        last = payload[-1]
        if not isinstance(last, dict):
            continue
        departure = None
        text = last.get("requestedDeparture")
        if text:
            try:
                departure = parsedate_to_datetime(str(text))
            except (TypeError, ValueError, IndexError, OverflowError):
                departure = None
        requested = last.get("kWhRequested")
        try:
            requested_f = float(requested) if requested is not None else None
        except (TypeError, ValueError):
            requested_f = None
        if requested_f is not None and (requested_f <= 0 or requested_f > 100):
            requested_f = None
        connection = row["connectionTime_utc"]
        if connection is not None and connection.tzinfo is None:
            connection = connection.replace(tzinfo=ZoneInfo("UTC"))
        delta = None
        if departure is not None and connection is not None:
            delta = departure - connection
            if delta <= timedelta(0) or delta > timedelta(hours=36):
                delta = None
        if delta is None and requested_f is None:
            continue
        found[_normalize_session_id(row["sessionID"])] = (delta, requested_f)
    return found


def _by_day(sessions: list[ScenarioSession]) -> dict:
    grouped: dict = defaultdict(list)
    for session in sessions:
        grouped[session.connection.astimezone(LA).date()].append(session)
    return grouped


def _offline_unmet(sessions: list[ScenarioSession], cap_kw: float, penalty: float) -> float:
    if not sessions:
        return 0.0
    start = min(session.connection for session in sessions)
    end = max(session.disconnect for session in sessions)
    origin = start.replace(minute=(start.minute // 15) * 15, second=0, microsecond=0)
    step = timedelta(minutes=15)
    slots: list[datetime] = []
    cursor = origin
    while cursor < end:
        slots.append(cursor)
        cursor += step
    if not slots:
        return sum(session.kwh for session in sessions)
    vehicles = [
        Vehicle(
            vehicle_id=session.session_id,
            energy_remaining_kwh=session.kwh,
            battery_max_kwh=session.kwh,
            window_start=session.connection,
            window_end=session.disconnect,
            p_max_kw=session.p_max_kw,
            arrival=session.connection,
        )
        for session in sessions
    ]
    prices = np.ones(len(slots))
    schedule = solve_epoch_fast(FleetState(vehicles), slots, prices, cap_kw, 0.25, penalty)
    return float(schedule.unmet_kwh.sum())


def min_cap_kw(sessions: list[ScenarioSession], penalty: float = 1000.0) -> tuple[float, float]:
    """Smallest site cap that delivers every kWh the plugs can physically deliver.

    Returns (cap_kw, kWh that stays unmet even at an unlimited cap).
    """
    if not sessions:
        return 0.0, 0.0
    hi = float(sum(session.p_max_kw for session in sessions) + 1.0)
    floor = _offline_unmet(sessions, hi, penalty)
    lo = 0.0
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        if _offline_unmet(sessions, mid, penalty) <= floor + 1e-3:
            hi = mid
        else:
            lo = mid
    return hi, floor


def scenario_min_cap(sessions: list[ScenarioSession], penalty: float = 1000.0) -> tuple[float, float]:
    """Max over days of the smallest cap that serves that day. Also the undeliverable kWh."""
    caps: list[float] = []
    undeliverable = 0.0
    for group in _by_day(sessions).values():
        cap, floor = min_cap_kw(group, penalty)
        caps.append(cap)
        undeliverable += floor
    return (max(caps) if caps else 0.0), undeliverable


def run_policy_year(
    sessions: list[ScenarioSession],
    policy,
    tier: str,
    policy_name: str,
    *,
    cap_kw: float,
    windows: list[dict],
    median_dwell: timedelta,
    penalty: float,
    demand_rate: float,
) -> dict:
    loop = MPCLoop(
        slot_minutes=15,
        committed_k=1,
        penalty_unmet=penalty,
        cap_kw=cap_kw,
        tariff_windows=windows,
        timezone_name=CLOCK,
        median_dwell=median_dwell,
    )
    energy = 0.0
    penalty_cost = 0.0
    unmet = 0.0
    requested = 0.0
    peaks: dict = {}
    day_seconds: list[float] = []
    for _day, group in sorted(_by_day(sessions).items()):
        requested += sum(session.kwh for session in group)
        started = time.perf_counter()
        log = loop.run(group, policy, tier, policy_name)
        day_seconds.append(time.perf_counter() - started)
        _total, day_energy, day_penalty = log.schedule.costs(penalty)
        energy += day_energy
        penalty_cost += day_penalty
        unmet += float(log.schedule.unmet_kwh.sum())
        peaks = merge_peaks(peaks, monthly_peaks(log.schedule.slot_starts, log.schedule.power_kw, CLOCK))
    demand = demand_cost(peaks, demand_rate)
    seconds = np.asarray(day_seconds, dtype=float)
    return {
        "total_cost": energy + penalty_cost + demand,
        "energy_cost": energy,
        "penalty_cost": penalty_cost,
        "demand_cost": demand,
        "unmet_kwh": unmet,
        "unmet_rate": (unmet / requested) if requested else 0.0,
        "peak_kw": max(peaks.values()) if peaks else 0.0,
        "solve_time_p95": float(np.quantile(seconds, 0.95)) if seconds.size else 0.0,
        "plug_binding_fraction": plug_binding_fraction(sessions, n_plugs=54),
    }


def evaluate_scenario(
    sessions: list[ScenarioSession],
    *,
    cap_kw: float,
    tariffs: list[dict],
    median_dwell: timedelta,
    penalty: float,
) -> list[dict]:
    """tariffs items: name, windows, demand_rate."""
    policies = (
        ("P0", charge_on_arrival, "T2"),
        ("P1", cheapest_slots_greedy, "T2"),
        ("P2b", earliest_deadline_first, "T2"),
        ("P3", _fast(penalty), "T2"),
        ("P4", _fast(penalty), "T3"),
    )
    cap_year, undeliverable = scenario_min_cap(sessions, penalty)
    rows: list[dict] = []
    for tariff in tariffs:
        results = {}
        for name, policy, tier in policies:
            results[name] = run_policy_year(
                sessions,
                policy,
                tier,
                name,
                cap_kw=cap_kw,
                windows=tariff["windows"],
                median_dwell=median_dwell,
                penalty=penalty,
                demand_rate=float(tariff["demand_rate"]),
            )
            results[name]["tier"] = tier
        baseline = results["P0"]["total_cost"]
        oracle = results["P4"]["total_cost"]
        oracle_energy = results["P4"]["energy_cost"] + results["P4"]["penalty_cost"]
        inside_gap = float("nan")
        if tariff["name"] == "sce_ev4":
            inside = [
                session
                for session in sessions
                if (session.requested_departure or (session.connection + median_dwell)) <= session.disconnect
            ]
            if inside:
                common = dict(
                    cap_kw=cap_kw,
                    windows=tariff["windows"],
                    median_dwell=median_dwell,
                    penalty=penalty,
                    demand_rate=float(tariff["demand_rate"]),
                )
                p3_inside = run_policy_year(inside, _fast(penalty), "T2", "P3", **common)
                p4_inside = run_policy_year(inside, _fast(penalty), "T3", "P4", **common)
                if p4_inside["total_cost"]:
                    inside_gap = (p3_inside["total_cost"] - p4_inside["total_cost"]) / p4_inside["total_cost"]
        for name, payload in results.items():
            payload["cost_reduction_vs_p0"] = (
                0.0 if name == "P0" or baseline == 0 else (baseline - payload["total_cost"]) / baseline
            )
            payload["optimality_gap_p3_p4"] = (
                (payload["total_cost"] - oracle) / oracle if name == "P3" and oracle else float("nan")
            )
            energy_bill = payload["energy_cost"] + payload["penalty_cost"]
            payload["gap_energy_only"] = (
                (energy_bill - oracle_energy) / oracle_energy if name == "P3" and oracle_energy else float("nan")
            )
            payload["gap_deadline_inside"] = inside_gap if name == "P3" else float("nan")
            payload["min_cap_kw"] = cap_year
            payload["undeliverable_kwh"] = undeliverable
            payload["tariff"] = tariff["name"]
            rows.append({"policy": name, **payload})
    return rows
