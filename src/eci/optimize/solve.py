"""Solve one LP epoch with scipy.optimize.milp (HiGHS, all variables continuous)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import vstack

from eci.optimize.formulation import build_lp
from eci.optimize.state import FleetState, Schedule


def solve_epoch(
    state: FleetState,
    slot_starts: list,
    prices: np.ndarray,
    cap_kw: float,
    dt_hours: float,
    penalty_unmet: float,
    peak_weight: float = 0.0,
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
            peak_duals=np.zeros(n_t),
        )
    built = build_lp(
        state,
        np.asarray(prices, dtype=float),
        cap_kw,
        dt_hours,
        penalty_unmet,
        slot_starts,
        peak_weight=peak_weight,
    )
    if peak_weight > 0:
        result, cap_duals, peak_duals = _linprog_with_duals(built)
    else:
        result = milp(
            built["c"],
            integrality=np.zeros(built["c"].shape[0]),
            bounds=Bounds(built["lb"], built["ub"]),
            constraints=LinearConstraint(built["matrix"], built["cons_lb"], built["cons_ub"]),
            options={"disp": False, "time_limit": 60},
        )
        cap_duals, peak_duals = _milp_duals(result, built)
    if not result.success or result.x is None:
        raise RuntimeError(f"LP epoch failed: {getattr(result, 'message', 'linprog')}")
    n_i = built["n_i"]
    n_t = built["n_t"]
    power = result.x[: n_i * n_t].reshape(n_i, n_t)
    unmet = result.x[n_i * n_t : n_i * n_t + n_i]
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
        peak_duals=peak_duals,
    )


def _milp_duals(result, built: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
    n_t = built["n_t"]
    n_i = built["n_i"]
    marginals = getattr(getattr(result, "ineqlin", None), "marginals", None)
    cap_duals = None
    peak_duals = None
    if marginals is not None and len(marginals) >= n_t:
        cap_duals = np.asarray(marginals[:n_t], dtype=float)
        peak_start = n_t + 2 * n_i
        if built.get("n_peak_rows") and len(marginals) >= peak_start + n_t:
            peak_duals = np.asarray(marginals[peak_start : peak_start + n_t], dtype=float)
    return cap_duals, peak_duals


def _linprog_with_duals(built: dict):
    """HiGHS linprog so cap duals exist when the peak epigraph is on."""
    matrix = built["matrix"].tocsr()
    blocks = []
    rhs = []
    cap_idx = []
    peak_idx = []
    n_t = built["n_t"]
    n_peak = int(built.get("n_peak_rows") or 0)
    peak_start = matrix.shape[0] - n_peak if n_peak else matrix.shape[0]
    row_out = 0
    for i in range(matrix.shape[0]):
        lo = built["cons_lb"][i]
        hi = built["cons_ub"][i]
        if np.isfinite(hi):
            if i < n_t:
                cap_idx.append(row_out)
            if n_peak and i >= peak_start:
                peak_idx.append(row_out)
            blocks.append(matrix.getrow(i))
            rhs.append(hi)
            row_out += 1
        if np.isfinite(lo):
            blocks.append(-matrix.getrow(i))
            rhs.append(-lo)
            row_out += 1
    a_ub = vstack(blocks, format="csr") if blocks else None
    result = linprog(
        built["c"],
        A_ub=a_ub,
        b_ub=np.asarray(rhs, dtype=float) if rhs else None,
        bounds=list(zip(built["lb"], built["ub"], strict=True)),
        method="highs",
    )
    marginals = getattr(getattr(result, "ineqlin", None), "marginals", None)
    cap_duals = np.zeros(n_t)
    peak_duals = np.zeros(n_t) if n_peak else None
    if marginals is not None:
        for slot, index in enumerate(cap_idx):
            cap_duals[slot] = float(marginals[index])
        if peak_duals is not None:
            for slot, index in enumerate(peak_idx):
                peak_duals[slot] = float(marginals[index])
    return result, cap_duals, peak_duals


def solve_epoch_fast(
    state: FleetState,
    slot_starts: list,
    prices: np.ndarray,
    cap_kw: float,
    dt_hours: float,
    penalty_unmet: float,
) -> Schedule:
    """Same objective as ``solve_epoch`` without an LP call.

    Slots are filled cheapest-first. On a slot, the vehicle with the fewest
    remaining feasible slots takes power first, so a car that still has a
    cheap slot does not consume the only slot of someone else. Unmet penalty
    is identical across vehicles, so who is short does not change the cost
    once the energy at each price is fixed.
    """
    n_t = len(slot_starts)
    n_i = len(state.vehicles)
    prices = np.asarray(prices, dtype=float)
    if n_i == 0 or n_t == 0:
        return solve_epoch(state, slot_starts, prices, cap_kw, dt_hours, penalty_unmet)
    slot_ts = np.array([slot.timestamp() for slot in slot_starts], dtype=float)
    feasible = np.zeros((n_i, n_t), dtype=bool)
    need = np.zeros(n_i)
    p_max = np.zeros(n_i)
    for i, vehicle in enumerate(state.vehicles):
        feasible[i] = (slot_ts >= vehicle.window_start.timestamp()) & (slot_ts < vehicle.window_end.timestamp())
        need[i] = min(max(vehicle.energy_remaining_kwh, 0.0), max(vehicle.battery_max_kwh, 0.0))
        p_max[i] = vehicle.p_max_kw
    power = np.zeros((n_i, n_t))
    for t in np.argsort(prices, kind="mergesort"):
        candidates = [i for i in range(n_i) if feasible[i, t] and need[i] > 1e-12]
        candidates.sort(key=lambda i: (int(feasible[i].sum()), i))
        left = cap_kw
        for i in candidates:
            if left <= 1e-12 or need[i] <= 1e-12:
                continue
            take = min(p_max[i], need[i] / dt_hours, left)
            if take <= 0:
                continue
            power[i, t] = take
            need[i] -= take * dt_hours
            left -= take
        feasible[:, t] = False
    unmet = np.maximum(need, 0.0)
    return Schedule(
        vehicle_ids=[vehicle.vehicle_id for vehicle in state.vehicles],
        slot_starts=list(slot_starts),
        power_kw=power,
        unmet_kwh=unmet,
        prices=prices,
        cap_kw=cap_kw,
        dt_hours=dt_hours,
        status="fast",
        cap_duals=np.zeros(n_t),
    )
