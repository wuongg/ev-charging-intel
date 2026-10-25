"""Independent check of a schedule. Solver status is not evidence."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from eci.optimize.state import FleetState, Schedule, Vehicle


class FeasibilityError(RuntimeError):
    def __init__(self, violations: list[str]):
        self.violations = violations
        super().__init__("; ".join(violations))


def _covers(slot: datetime, vehicle: Vehicle) -> bool:
    return vehicle.window_start <= slot < vehicle.window_end


def check_feasibility(
    schedule: Schedule,
    state: FleetState,
    cap_kw: float | None = None,
    tol: float = 1e-4,
) -> list[str]:
    """Recompute cap, window, power limit, battery, and energy balance from the rates."""
    violations: list[str] = []
    cap = schedule.cap_kw if cap_kw is None else cap_kw
    power = np.asarray(schedule.power_kw, dtype=float)
    unmet = np.asarray(schedule.unmet_kwh, dtype=float)
    n_i = len(state.vehicles)
    n_t = len(schedule.slot_starts)
    if power.shape != (n_i, n_t):
        return [f"power shape {power.shape} != {(n_i, n_t)}"]
    if unmet.shape != (n_i,):
        return [f"unmet shape {unmet.shape} != {(n_i,)}"]
    if np.any(power < -tol):
        violations.append("negative power")
    if np.any(unmet < -tol):
        violations.append("negative unmet")
    column = power.sum(axis=0)
    if np.any(column > cap + tol):
        violations.append(f"cap exceeded at {int(np.argmax(column > cap + tol))}")
    for index, vehicle in enumerate(state.vehicles):
        if np.any(power[index] > vehicle.p_max_kw + tol):
            violations.append(f"{vehicle.vehicle_id} exceeds p_max")
        for t, slot in enumerate(schedule.slot_starts):
            if not _covers(slot, vehicle) and power[index, t] > tol:
                violations.append(f"{vehicle.vehicle_id} power outside window at {slot.isoformat()}")
                break
        delivered = float(power[index].sum() * schedule.dt_hours)
        if delivered > vehicle.battery_max_kwh + tol:
            violations.append(f"{vehicle.vehicle_id} exceeds battery")
        required = max(vehicle.energy_remaining_kwh, 0.0)
        if delivered + float(unmet[index]) + tol < required:
            violations.append(f"{vehicle.vehicle_id} energy below requirement")
    return violations


def assert_feasible(schedule: Schedule, state: FleetState, cap_kw: float | None = None, tol: float = 1e-4) -> None:
    violations = check_feasibility(schedule, state, cap_kw=cap_kw, tol=tol)
    if violations:
        raise FeasibilityError(violations)
