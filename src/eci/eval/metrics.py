"""Scenario aggregates. Costs add the monthly demand charge on top of the LP energy bill."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from eci.optimize.mpc import ScenarioSession
from eci.optimize.state import Schedule


def plug_binding_fraction(sessions: list[ScenarioSession], n_plugs: int, slot_minutes: int = 15) -> float:
    """Share of 15-minute slots in which the plug count meets the stall count."""
    if not sessions or n_plugs <= 0:
        return 0.0
    start = min(session.connection for session in sessions)
    end = max(session.disconnect for session in sessions)
    origin = start.replace(minute=(start.minute // slot_minutes) * slot_minutes, second=0, microsecond=0)
    step = slot_minutes * 60
    n_slots = int(np.ceil((end - origin).total_seconds() / step)) + 1
    if n_slots <= 0:
        return 0.0
    diff = np.zeros(n_slots + 1, dtype=np.int32)
    for session in sessions:
        a = int((session.connection - origin).total_seconds() // step)
        b = int(np.ceil((session.disconnect - origin).total_seconds() / step))
        a = min(max(a, 0), n_slots)
        b = min(max(b, 0), n_slots)
        diff[a] += 1
        diff[b] -= 1
    occupied = np.cumsum(diff[:-1])
    return float(np.mean(occupied >= n_plugs))


def monthly_peaks(
    slot_starts: list[datetime], power_kw: np.ndarray, timezone_name: str
) -> dict[tuple[int, int], float]:
    """Max metered kW in each local (year, month)."""
    zone = ZoneInfo(timezone_name)
    column = np.asarray(power_kw, dtype=float).sum(axis=0)
    peaks: dict[tuple[int, int], float] = {}
    for slot, value in zip(slot_starts, column, strict=True):
        local = slot.astimezone(zone)
        key = (local.year, local.month)
        peaks[key] = max(peaks.get(key, 0.0), float(value))
    return peaks


def demand_cost(peaks: dict[tuple[int, int], float], rate_per_kw_month: float) -> float:
    return float(rate_per_kw_month) * float(sum(peaks.values()))


def merge_peaks(*groups: dict[tuple[int, int], float]) -> dict[tuple[int, int], float]:
    merged: dict[tuple[int, int], float] = {}
    for group in groups:
        for key, value in group.items():
            merged[key] = max(merged.get(key, 0.0), value)
    return merged


def cost_reduction(baseline: float, other: float) -> float:
    if baseline == 0:
        return float("nan")
    return (baseline - other) / baseline


def optimality_gap(policy_cost: float, oracle_cost: float) -> float:
    if oracle_cost == 0:
        return float("nan")
    return (policy_cost - oracle_cost) / oracle_cost


def schedule_energy_and_penalty(schedule: Schedule, penalty: float) -> tuple[float, float]:
    _total, energy, penalty_cost = schedule.costs(penalty)
    return energy, penalty_cost
