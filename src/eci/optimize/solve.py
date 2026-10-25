"""Solve one LP epoch with scipy.optimize.milp (HiGHS, all variables continuous)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from eci.optimize.formulation import build_lp
from eci.optimize.state import FleetState, Schedule


def solve_epoch(
    state: FleetState,
    slot_starts: list,
    prices: np.ndarray,
    cap_kw: float,
    dt_hours: float,
    penalty_unmet: float,
) -> Schedule:
    """Rates, status, and cap-row duals. An empty fleet is a zero schedule."""
    n_t = len(slot_starts)
    if not state.vehicles or n_t == 0:
        return Schedule(
            vehicle_ids=[vehicle.vehicle_id for vehicle in state.vehicles],
            slot_starts=list(slot_starts),
            power_kw=np.zeros((len(state.vehicles), n_t)),
            unmet_kwh=np.array([vehicle.energy_remaining_kwh for vehicle in state.vehicles]),
            prices=np.asarray(prices, dtype=float),
            cap_kw=cap_kw,
            dt_hours=dt_hours,
            status="empty",
            cap_duals=np.zeros(n_t),
        )
    built = build_lp(state, np.asarray(prices, dtype=float), cap_kw, dt_hours, penalty_unmet, slot_starts)
    result = milp(
        built["c"],
        integrality=np.zeros(built["c"].shape[0]),
        bounds=Bounds(built["lb"], built["ub"]),
        constraints=LinearConstraint(built["matrix"], built["cons_lb"], built["cons_ub"]),
        options={"disp": False, "time_limit": 60},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"LP epoch failed: {result.message}")
    n_i = built["n_i"]
    n_t = built["n_t"]
    power = result.x[: n_i * n_t].reshape(n_i, n_t)
    unmet = result.x[n_i * n_t :]
    marginals = getattr(getattr(result, "ineqlin", None), "marginals", None)
    cap_duals = None
    if marginals is not None and len(marginals) >= n_t:
        cap_duals = np.asarray(marginals[:n_t], dtype=float)
    return Schedule(
        vehicle_ids=[vehicle.vehicle_id for vehicle in state.vehicles],
        slot_starts=list(slot_starts),
        power_kw=np.maximum(power, 0.0),
        unmet_kwh=np.maximum(unmet, 0.0),
        prices=np.asarray(prices, dtype=float),
        cap_kw=cap_kw,
        dt_hours=dt_hours,
        status=str(result.message),
        cap_duals=cap_duals,
    )
