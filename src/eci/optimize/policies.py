"""P0 charges as soon as a vehicle is plugged. P1 fills the cheapest feasible slots."""

from __future__ import annotations

import numpy as np

from eci.optimize.state import FleetState, Schedule


def _blank(state: FleetState, slot_starts: list, prices: np.ndarray, cap_kw: float, dt_hours: float) -> np.ndarray:
    return np.zeros((len(state.vehicles), len(slot_starts)))


def _finish(
    state: FleetState,
    power: np.ndarray,
    slot_starts: list,
    prices,
    cap_kw: float,
    dt_hours: float,
    name: str,
) -> Schedule:
    delivered = power.sum(axis=1) * dt_hours
    remaining = np.array([max(vehicle.energy_remaining_kwh, 0.0) for vehicle in state.vehicles])
    unmet = np.maximum(remaining - delivered, 0.0)
    return Schedule(
        vehicle_ids=[vehicle.vehicle_id for vehicle in state.vehicles],
        slot_starts=list(slot_starts),
        power_kw=power,
        unmet_kwh=unmet,
        prices=np.asarray(prices, dtype=float),
        cap_kw=cap_kw,
        dt_hours=dt_hours,
        status=name,
    )


def charge_on_arrival(
    state: FleetState,
    slot_starts: list,
    prices: np.ndarray,
    cap_kw: float,
    dt_hours: float,
) -> Schedule:
    """Max power immediately, truncated by the cap in arrival order (FIFO)."""
    return _fill_now(state, slot_starts, prices, cap_kw, dt_hours, by_deadline=False)


def earliest_deadline_first(
    state: FleetState,
    slot_starts: list,
    prices: np.ndarray,
    cap_kw: float,
    dt_hours: float,
) -> Schedule:
    """P2b. Each slot serves the earliest deadline first, up to the cap."""
    return _fill_now(state, slot_starts, prices, cap_kw, dt_hours, by_deadline=True)


def _fill_now(state, slot_starts, prices, cap_kw, dt_hours, by_deadline: bool) -> Schedule:
    power = _blank(state, slot_starts, prices, cap_kw, dt_hours)
    if by_deadline:
        order = sorted(range(len(state.vehicles)), key=lambda index: (state.vehicles[index].window_end, index))
        name = "earliest_deadline_first"
    else:
        order = sorted(range(len(state.vehicles)), key=lambda index: (state.vehicles[index].arrival, index))
        name = "charge_on_arrival"
    need = [max(vehicle.energy_remaining_kwh, 0.0) for vehicle in state.vehicles]
    for t, slot in enumerate(slot_starts):
        left = cap_kw
        for index in order:
            vehicle = state.vehicles[index]
            if need[index] <= 1e-9 or not (vehicle.window_start <= slot < vehicle.window_end):
                continue
            take = min(vehicle.p_max_kw, need[index] / dt_hours, left)
            if take <= 0:
                continue
            power[index, t] = take
            need[index] -= take * dt_hours
            left -= take
    return _finish(state, power, slot_starts, prices, cap_kw, dt_hours, name)


def cheapest_slots_greedy(
    state: FleetState,
    slot_starts: list,
    prices: np.ndarray,
    cap_kw: float,
    dt_hours: float,
) -> Schedule:
    """Earliest-deadline vehicle first, then the cheapest open slots in its window."""
    power = _blank(state, slot_starts, prices, cap_kw, dt_hours)
    cap_left = np.full(len(slot_starts), cap_kw, dtype=float)
    prices = np.asarray(prices, dtype=float)
    order = sorted(range(len(state.vehicles)), key=lambda index: (state.vehicles[index].window_end, index))
    for index in order:
        vehicle = state.vehicles[index]
        need = max(vehicle.energy_remaining_kwh, 0.0)
        eligible = [
            t
            for t, slot in enumerate(slot_starts)
            if vehicle.window_start <= slot < vehicle.window_end
        ]
        eligible.sort(key=lambda t: (prices[t], slot_starts[t]))
        for t in eligible:
            if need <= 1e-9:
                break
            take = min(vehicle.p_max_kw, need / dt_hours, cap_left[t])
            if take <= 0:
                continue
            power[index, t] = take
            need -= take * dt_hours
            cap_left[t] -= take
    return _finish(state, power, slot_starts, prices, cap_kw, dt_hours, "cheapest_slots_greedy")
