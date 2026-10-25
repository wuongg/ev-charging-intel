"""LP optimum matches an enumerated assignment on a 3-vehicle, 4-slot case."""

from datetime import datetime, timezone

import numpy as np
import pytest

from eci.optimize.policies import charge_on_arrival
from eci.optimize.solve import solve_epoch, solve_epoch_fast
from eci.optimize.state import FleetState, Vehicle


def _vehicles() -> list[Vehicle]:
    start = datetime(2019, 1, 8, 0, tzinfo=timezone.utc)
    end = datetime(2019, 1, 8, 4, tzinfo=timezone.utc)
    return [
        Vehicle(
            vehicle_id=f"ev{index}",
            energy_remaining_kwh=1.0,
            battery_max_kwh=1.0,
            window_start=start,
            window_end=end,
            p_max_kw=1.0,
            arrival=start,
        )
        for index in range(3)
    ]


def _brute_min_cost(prices: np.ndarray, cap: int = 2) -> float:
    best = None
    for a in range(4):
        for b in range(4):
            for c in range(4):
                counts = [0, 0, 0, 0]
                for choice in (a, b, c):
                    counts[choice] += 1
                if max(counts) > cap:
                    continue
                cost = float(prices[a] + prices[b] + prices[c])
                best = cost if best is None else min(best, cost)
    return float(best)


def test_lp_matches_enumerated_optimum() -> None:
    start = datetime(2019, 1, 8, 0, tzinfo=timezone.utc)
    slots = [start.replace(hour=hour) for hour in range(4)]
    prices = np.array([5.0, 1.0, 5.0, 1.0])
    state = FleetState(_vehicles())
    schedule = solve_epoch(state, slots, prices, cap_kw=2.0, dt_hours=1.0, penalty_unmet=1000.0)
    total, energy, penalty = schedule.costs(1000.0)
    assert penalty == 0.0
    assert energy == _brute_min_cost(prices)
    assert abs(total - energy) < 1e-6


def test_charge_on_arrival_is_not_the_only_feasible_point() -> None:
    start = datetime(2019, 1, 8, 0, tzinfo=timezone.utc)
    slots = [start.replace(hour=hour) for hour in range(4)]
    prices = np.array([5.0, 1.0, 5.0, 1.0])
    state = FleetState(_vehicles())
    immediate = charge_on_arrival(state, slots, prices, cap_kw=2.0, dt_hours=1.0)
    optimal = solve_epoch(state, slots, prices, cap_kw=2.0, dt_hours=1.0, penalty_unmet=1000.0)
    assert immediate.costs(1000.0)[0] >= optimal.costs(1000.0)[0] - 1e-6


def test_fast_filler_matches_lp_cost() -> None:
    start = datetime(2019, 1, 8, 0, tzinfo=timezone.utc)
    slots = [start.replace(hour=hour) for hour in range(4)]
    prices = np.array([5.0, 1.0, 5.0, 1.0])
    state = FleetState(_vehicles())
    lp = solve_epoch(state, slots, prices, cap_kw=2.0, dt_hours=1.0, penalty_unmet=1000.0)
    fast = solve_epoch_fast(state, slots, prices, cap_kw=2.0, dt_hours=1.0, penalty_unmet=1000.0)
    assert fast.costs(1000.0)[0] == pytest.approx(lp.costs(1000.0)[0])
