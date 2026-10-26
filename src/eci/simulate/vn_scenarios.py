"""Vietnam scenario axes. S1 is an assumed Bernoulli grid-outage process.

Parameters are labeled assumptions calibrated to public reports of VN grid
events (2023 Northern burnout, 2024 Yagi, 2025-26 peak records). They are not
measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import yaml

from eci.config import REPO_ROOT
from eci.eval.dispatch import _by_day, demand_cost, merge_peaks, monthly_peaks, run_policy_year
from eci.eval.invariants import check_invariants
from eci.optimize.mpc import MPCLoop, ScenarioSession, _floor_slot
from eci.optimize.policies import charge_on_arrival
from eci.optimize.solve import solve_epoch_fast

LA = ZoneInfo("America/Los_Angeles")
DRY_MONTHS = frozenset({11, 12, 1, 2, 3, 4})


@dataclass(frozen=True)
class Outage:
    start: datetime
    end: datetime


def load_s1(path=None) -> dict:
    raw = yaml.safe_load((path or (REPO_ROOT / "configs" / "scenarios.yaml")).read_text(encoding="utf-8"))
    s1 = dict(raw.get("S1") or {})
    if not s1:
        raise ValueError("configs/scenarios.yaml is missing S1")
    return s1


def is_dry(day: date) -> bool:
    return day.month in DRY_MONTHS


def sample_outages(
    days: list[date],
    params: dict,
    *,
    seed: int,
    timezone_name: str = "America/Los_Angeles",
) -> list[Outage]:
    """Per-day Bernoulli, duration lognormal clipped to 1–6 h, peak-hour start bias."""
    rng = np.random.default_rng(int(seed))
    zone = ZoneInfo(timezone_name)
    p_dry = float(params["p_dry"])
    p_wet = float(params["p_wet"])
    mu = float(params["dur_lognorm_mu"])
    sigma = float(params["dur_lognorm_sigma"])
    lo = float(params.get("duration_min_h", 1.0))
    hi = float(params.get("duration_max_h", 6.0))
    multiplier = float(params["peak_hour_multiplier"])
    peak = set(int(hour) for hour in params.get("peak_hours", [17, 18, 19, 20, 21]))
    hours = np.arange(24)
    weights = np.array([multiplier if int(hour) in peak else 1.0 for hour in hours], dtype=float)
    weights /= weights.sum()
    outages: list[Outage] = []
    for day in days:
        p = p_dry if is_dry(day) else p_wet
        if rng.random() >= p:
            continue
        hour = int(rng.choice(hours, p=weights))
        minute = int(rng.integers(0, 4)) * 15
        start = datetime(day.year, day.month, day.day, hour, minute, tzinfo=zone)
        duration = float(np.clip(rng.lognormal(mu, sigma), lo, hi))
        outages.append(Outage(start=start, end=start + timedelta(hours=duration)))
    return outages


def slot_in_outage(slot: datetime, outages: list[Outage]) -> bool:
    return any(event.start <= slot < event.end for event in outages)


def outage_hook_factory(slots: list[datetime], outages: list[Outage], dt_hours: float):
    """Zero plant delivery on outage epochs. The policy is not told in advance."""

    def hook(epoch, active_index, delivered, requirements, committed):
        if epoch >= len(slots) or not slot_in_outage(slots[epoch], outages):
            return 0.0
        for index in active_index:
            delivered[index] -= float(committed[index, epoch]) * dt_hours
            committed[index, epoch] = 0.0
        return 0.0

    return hook


def _loop(cap_kw: float, windows: list[dict], penalty: float, median_dwell: timedelta) -> MPCLoop:
    return MPCLoop(
        slot_minutes=15,
        committed_k=1,
        penalty_unmet=penalty,
        cap_kw=cap_kw,
        tariff_windows=windows,
        timezone_name="America/Los_Angeles",
        median_dwell=median_dwell,
    )


def _p3(penalty: float):
    def policy(state, slots, prices, cap, dt):
        return solve_epoch_fast(state, slots, prices, cap, dt, penalty)

    return policy


def energy_in_outage(sessions: list[ScenarioSession], outages: list[Outage]) -> float:
    """kWh whose physical deadline falls inside an outage — irrecoverable by re-solve."""
    total = 0.0
    for session in sessions:
        if any(event.start <= session.disconnect <= event.end for event in outages):
            total += session.kwh
    return total


def outage_hours_overlapping(sessions: list[ScenarioSession], outages: list[Outage]) -> float:
    hours = 0.0
    for event in outages:
        cursor = event.start
        step = timedelta(minutes=15)
        while cursor < event.end:
            if any(session.connection <= cursor < session.disconnect for session in sessions):
                hours += 0.25
            cursor += step
    return hours


def run_policy_with_outage(
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
    outages: list[Outage] | None,
) -> dict:
    if not outages:
        result = run_policy_year(
            sessions,
            policy,
            tier,
            policy_name,
            cap_kw=cap_kw,
            windows=windows,
            median_dwell=median_dwell,
            penalty=penalty,
            demand_rate=demand_rate,
        )
        result["i1_ok"] = True
        result["unmet_in_outage"] = 0.0
        result["outage_hours_served_missed"] = 0.0
        return result

    energy = 0.0
    penalty_cost = 0.0
    unmet = 0.0
    requested = 0.0
    peaks: dict = {}
    i1_ok = True
    for _day, group in sorted(_by_day(sessions).items()):
        requested += sum(session.kwh for session in group)
        start = _floor_slot(min(session.connection for session in group), 15)
        end = max(session.disconnect for session in group)
        slots: list[datetime] = []
        cursor = start
        while cursor < end:
            slots.append(cursor)
            cursor += timedelta(minutes=15)
        hook = outage_hook_factory(slots, outages, 0.25)
        log = _loop(cap_kw, windows, penalty, median_dwell).run(
            group, policy, tier, policy_name, plant_hook=hook
        )
        violations = check_invariants(log.schedule, group)
        if any(item.startswith("I1") for item in violations):
            i1_ok = False
        _total, day_energy, day_penalty = log.schedule.costs(penalty)
        energy += day_energy
        penalty_cost += day_penalty
        unmet += float(log.schedule.unmet_kwh.sum())
        peaks = merge_peaks(
            peaks, monthly_peaks(log.schedule.slot_starts, log.schedule.power_kw, "America/Los_Angeles")
        )
    demand = demand_cost(peaks, demand_rate)
    return {
        "total_cost": energy + penalty_cost + demand,
        "energy_cost": energy,
        "penalty_cost": penalty_cost,
        "demand_cost": demand,
        "unmet_kwh": unmet,
        "unmet_rate": (unmet / requested) if requested else 0.0,
        "peak_kw": max(peaks.values()) if peaks else 0.0,
        "i1_ok": i1_ok,
        "unmet_in_outage": energy_in_outage(sessions, outages),
        "outage_hours_served_missed": outage_hours_overlapping(sessions, outages),
    }


def compare_outage(
    sessions: list[ScenarioSession],
    *,
    cap_kw: float,
    windows: list[dict],
    median_dwell: timedelta,
    penalty: float,
    demand_rate: float,
    outages: list[Outage],
) -> list[dict]:
    policies = (
        ("P0", charge_on_arrival, "T3"),
        ("P3", _p3(penalty), "T3"),
    )
    rows = []
    for name, policy, tier in policies:
        off = run_policy_with_outage(
            sessions,
            policy,
            tier,
            name,
            cap_kw=cap_kw,
            windows=windows,
            median_dwell=median_dwell,
            penalty=penalty,
            demand_rate=demand_rate,
            outages=None,
        )
        on = run_policy_with_outage(
            sessions,
            policy,
            tier,
            name,
            cap_kw=cap_kw,
            windows=windows,
            median_dwell=median_dwell,
            penalty=penalty,
            demand_rate=demand_rate,
            outages=outages,
        )
        recovery = max(float(on["energy_cost"]) - float(off["energy_cost"]), 0.0)
        rows.append(
            {
                "scenario": f"{name}_outage_off",
                "outage_hours_served_missed": 0.0,
                "cost": off["total_cost"],
                "unmet_in_outage": 0.0,
                "recovery_cost": 0.0,
                "unmet_kwh": off["unmet_kwh"],
                "unmet_rate": off["unmet_rate"],
                "peak_kw": off["peak_kw"],
                "i1_ok": off["i1_ok"],
            }
        )
        rows.append(
            {
                "scenario": f"{name}_outage_on",
                "outage_hours_served_missed": on["outage_hours_served_missed"],
                "cost": on["total_cost"],
                "unmet_in_outage": on["unmet_in_outage"],
                "recovery_cost": recovery,
                "unmet_kwh": on["unmet_kwh"],
                "unmet_rate": on["unmet_rate"],
                "peak_kw": on["peak_kw"],
                "i1_ok": on["i1_ok"],
            }
        )
    return rows
