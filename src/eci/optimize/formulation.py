"""Time-indexed LP. Continuous power and unmet energy. No binaries, no peak epigraph."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from scipy.sparse import lil_matrix

from eci.optimize.state import FleetState

_DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def slot_price(moment: datetime, windows: list[dict], timezone_name: str) -> float:
    """Placeholder tariff price for the local clock minute that starts ``moment``."""
    local = moment.astimezone(ZoneInfo(timezone_name))
    day = _DAYS[local.weekday()]
    minute = local.hour * 60 + local.minute
    for window in windows:
        if day not in window["days"]:
            continue
        start_h, start_m = (int(part) for part in str(window["start"]).split(":"))
        end_h, end_m = (int(part) for part in str(window["end"]).split(":"))
        start = start_h * 60 + start_m
        end = end_h * 60 + end_m
        if start <= minute < end:
            return float(window["price_per_kwh"])
    raise ValueError(f"no tariff window covers {local.isoformat()}")


def _in_window(slot: datetime, start: datetime, end: datetime) -> bool:
    return start <= slot < end


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
    for i, vehicle in enumerate(vehicles):
        for t, slot in enumerate(slot_starts):
            index = i * n_t + t
            c[index] = prices[t] * dt_hours
            if _in_window(slot, vehicle.window_start, vehicle.window_end):
                ub[index] = vehicle.p_max_kw
            else:
                ub[index] = 0.0
        unmet = n_i * n_t + i
        c[unmet] = penalty_unmet
        ub[unmet] = max(vehicle.energy_remaining_kwh, 0.0)
    n_con = n_t + 2 * n_i
    matrix = lil_matrix((n_con, n_var))
    cons_lb = np.full(n_con, -np.inf)
    cons_ub = np.zeros(n_con)
    for t in range(n_t):
        for i in range(n_i):
            matrix[t, i * n_t + t] = 1.0
        cons_ub[t] = cap_kw
    for i, vehicle in enumerate(vehicles):
        energy_row = n_t + i
        battery_row = n_t + n_i + i
        for t in range(n_t):
            matrix[energy_row, i * n_t + t] = dt_hours
            matrix[battery_row, i * n_t + t] = dt_hours
        matrix[energy_row, n_i * n_t + i] = 1.0
        cons_lb[energy_row] = max(vehicle.energy_remaining_kwh, 0.0)
        cons_ub[energy_row] = np.inf
        cons_ub[battery_row] = max(vehicle.battery_max_kwh, 0.0)
    return {
        "c": c,
        "lb": lb,
        "ub": ub,
        "matrix": matrix.tocsr(),
        "cons_lb": cons_lb,
        "cons_ub": cons_ub,
        "n_cap_rows": n_t,
        "n_i": n_i,
        "n_t": n_t,
    }
