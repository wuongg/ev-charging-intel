"""Min-cost flow with the same TU structure as ADR-002. Optional OR-Tools extra."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from eci.optimize.state import FleetState, Schedule

# Integer milli-watt-hours. Unit costs are scaled so kWh * price reconstructs.
WH = 1_000_000
COST_SCALE = 10_000_000_000


def _require_ortools():
    try:
        from ortools.graph.python import min_cost_flow
    except ImportError as exc:
        raise ImportError("install the optional extra: pip install 'eci[flow]'") from exc
    return min_cost_flow


def solve_epoch_flow(
    state: FleetState,
    slot_starts: list[datetime],
    prices: np.ndarray,
    cap_kw: float,
    dt_hours: float,
    penalty_unmet: float,
) -> Schedule:
    """EV i supplies energy, slot t is a capacitated sink, arc (i, t) in the window."""
    min_cost_flow = _require_ortools()
    n_i = len(state.vehicles)
    n_t = len(slot_starts)
    prices = np.asarray(prices, dtype=float)
    if n_i == 0 or n_t == 0:
        return Schedule(
            vehicle_ids=[vehicle.vehicle_id for vehicle in state.vehicles],
            slot_starts=list(slot_starts),
            power_kw=np.zeros((n_i, n_t)),
            unmet_kwh=np.array([vehicle.energy_remaining_kwh for vehicle in state.vehicles]),
            prices=prices,
            cap_kw=cap_kw,
            dt_hours=dt_hours,
            status="empty",
            cap_duals=np.zeros(n_t),
        )
    slot_ts = np.array([slot.timestamp() for slot in slot_starts], dtype=float)
    sink = n_i + n_t
    unmet_node = sink + 1
    mcf = min_cost_flow.SimpleMinCostFlow()
    supplies = []
    for i, vehicle in enumerate(state.vehicles):
        energy = min(max(vehicle.energy_remaining_kwh, 0.0), max(vehicle.battery_max_kwh, 0.0))
        supply = int(round(energy * WH))
        supplies.append(supply)
        mcf.set_node_supply(i, supply)
    total = int(sum(supplies))
    mcf.set_node_supply(sink, -total)
    arcs: list[tuple[int, int, int]] = []
    for i, vehicle in enumerate(state.vehicles):
        inside = (slot_ts >= vehicle.window_start.timestamp()) & (slot_ts < vehicle.window_end.timestamp())
        cap_wh = int(round(vehicle.p_max_kw * dt_hours * WH))
        for t in range(n_t):
            if not inside[t] or cap_wh <= 0:
                continue
            cost = int(round(prices[t] * (COST_SCALE / WH)))
            arc = mcf.add_arc_with_capacity_and_unit_cost(i, n_i + t, cap_wh, cost)
            arcs.append((arc, i, t))
        unmet_cost = int(round(penalty_unmet * (COST_SCALE / WH)))
        mcf.add_arc_with_capacity_and_unit_cost(i, unmet_node, supplies[i], unmet_cost)
    slot_cap = int(round(cap_kw * dt_hours * WH))
    for t in range(n_t):
        mcf.add_arc_with_capacity_and_unit_cost(n_i + t, sink, slot_cap, 0)
    mcf.add_arc_with_capacity_and_unit_cost(unmet_node, sink, total, 0)
    status = mcf.solve()
    if status != mcf.OPTIMAL:
        raise RuntimeError(f"min-cost flow failed with status {status}")
    power = np.zeros((n_i, n_t))
    for arc, i, t in arcs:
        power[i, t] = mcf.flow(arc) / (WH * dt_hours) if dt_hours else 0.0
    delivered = power.sum(axis=1) * dt_hours
    remaining = np.array(
        [min(max(v.energy_remaining_kwh, 0.0), max(v.battery_max_kwh, 0.0)) for v in state.vehicles]
    )
    unmet = np.maximum(remaining - delivered, 0.0)
    # Integer supplies leave sub-unit dust; do not price that dust at M.
    unmet[unmet * WH < 2.0] = 0.0
    return Schedule(
        vehicle_ids=[vehicle.vehicle_id for vehicle in state.vehicles],
        slot_starts=list(slot_starts),
        power_kw=power,
        unmet_kwh=unmet,
        prices=prices,
        cap_kw=cap_kw,
        dt_hours=dt_hours,
        status="flow",
        cap_duals=np.zeros(n_t),
    )
