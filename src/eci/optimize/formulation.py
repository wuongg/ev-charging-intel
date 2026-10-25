"""Time-indexed LP. Continuous power and unmet energy. No binaries, no peak epigraph."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from scipy.sparse import csr_matrix

from eci.optimize.state import FleetState

_DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def slot_price(moment: datetime, windows: list[dict], timezone_name: str) -> float:
    """Energy price for the local clock minute that starts ``moment``.

    A window may list ``months`` (1–12). Windows without that key cover every month.
    """
    local = moment.astimezone(ZoneInfo(timezone_name))
    day = _DAYS[local.weekday()]
    minute = local.hour * 60 + local.minute
    for window in windows:
        months = window.get("months")
        if months is not None and local.month not in months:
            continue
        if day not in window["days"]:
            continue
        start_h, start_m = (int(part) for part in str(window["start"]).split(":"))
        end_h, end_m = (int(part) for part in str(window["end"]).split(":"))
        start = start_h * 60 + start_m
        end = end_h * 60 + end_m
        if start <= minute < end:
            return float(window["price_per_kwh"])
    raise ValueError(f"no tariff window covers {local.isoformat()}")


def build_lp(
    state: FleetState,
    prices: np.ndarray,
    cap_kw: float,
    dt_hours: float,
    penalty_unmet: float,
    slot_starts: list[datetime],
) -> dict:
    """Decision vector is p[i, t] then unmet[i].

    Cap rows come first so their duals are the leading inequality marginals.
    """
    vehicles = state.vehicles
    n_i = len(vehicles)
    n_t = len(slot_starts)
    n_var = n_i * n_t + n_i
    prices = np.asarray(prices, dtype=float)
    if prices.shape != (n_t,):
        raise ValueError("prices must have one entry per slot")
    c = np.zeros(n_var)
    lb = np.zeros(n_var)
    ub = np.zeros(n_var)
    slot_ts = np.array([slot.timestamp() for slot in slot_starts], dtype=float)
    for i, vehicle in enumerate(vehicles):
        index = slice(i * n_t, (i + 1) * n_t)
        c[index] = prices * dt_hours
        inside = (slot_ts >= vehicle.window_start.timestamp()) & (slot_ts < vehicle.window_end.timestamp())
        ub[index] = np.where(inside, vehicle.p_max_kw, 0.0)
        unmet = n_i * n_t + i
        c[unmet] = penalty_unmet
        ub[unmet] = max(vehicle.energy_remaining_kwh, 0.0)
    n_con = n_t + 2 * n_i
    cons_lb = np.full(n_con, -np.inf)
    cons_ub = np.zeros(n_con)
    cons_ub[:n_t] = cap_kw
    # Cap, energy, unmet column, battery. Built as index arrays; lil __setitem__ dominated the epoch.
    n_cap = n_i * n_t
    nnz = n_cap + n_cap + n_i + n_cap
    rows = np.empty(nnz, dtype=np.int32)
    cols = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=float)
    t_idx = np.tile(np.arange(n_t, dtype=np.int32), n_i)
    col_idx = (np.repeat(np.arange(n_i, dtype=np.int32), n_t) * n_t + np.tile(np.arange(n_t, dtype=np.int32), n_i))
    rows[:n_cap] = t_idx
    cols[:n_cap] = col_idx
    data[:n_cap] = 1.0
    energy_rows = n_t + np.repeat(np.arange(n_i, dtype=np.int32), n_t)
    battery_rows = n_t + n_i + np.repeat(np.arange(n_i, dtype=np.int32), n_t)
    rows[n_cap : 2 * n_cap] = energy_rows
    cols[n_cap : 2 * n_cap] = col_idx
    data[n_cap : 2 * n_cap] = dt_hours
    rows[2 * n_cap : 2 * n_cap + n_i] = n_t + np.arange(n_i, dtype=np.int32)
    cols[2 * n_cap : 2 * n_cap + n_i] = n_i * n_t + np.arange(n_i, dtype=np.int32)
    data[2 * n_cap : 2 * n_cap + n_i] = 1.0
    base = 2 * n_cap + n_i
    rows[base:] = battery_rows
    cols[base:] = col_idx
    data[base:] = dt_hours
    for i, vehicle in enumerate(vehicles):
        energy_row = n_t + i
        battery_row = n_t + n_i + i
        cons_lb[energy_row] = max(vehicle.energy_remaining_kwh, 0.0)
        cons_ub[energy_row] = np.inf
        cons_ub[battery_row] = max(vehicle.battery_max_kwh, 0.0)
    return {
        "c": c,
        "lb": lb,
        "ub": ub,
        "matrix": csr_matrix((data, (rows, cols)), shape=(n_con, n_var)),
        "cons_lb": cons_lb,
        "cons_ub": cons_ub,
        "n_cap_rows": n_t,
        "n_i": n_i,
        "n_t": n_t,
    }
