"""Peak weight in the LP lowers the realized peak on a toy fleet."""

from datetime import datetime, timezone

import numpy as np

from eci.optimize.solve import solve_epoch
from eci.optimize.state import FleetState, Vehicle


def test_peak_weight_lowers_peak() -> None:
    start = datetime(2019, 1, 8, 0, tzinfo=timezone.utc)
    slots = [start.replace(hour=hour) for hour in range(4)]
    prices = np.array([1.0, 1.0, 1.0, 1.0])
    vehicles = [
        Vehicle(
            vehicle_id=f"ev{index}",
            energy_remaining_kwh=1.0,
            battery_max_kwh=1.0,
            window_start=start,
            window_end=datetime(2019, 1, 8, 4, tzinfo=timezone.utc),
            p_max_kw=1.0,
            arrival=start,
        )
        for index in range(3)
    ]
    state = FleetState(vehicles)
    cheap = solve_epoch(state, slots, prices, cap_kw=3.0, dt_hours=1.0, penalty_unmet=25.0, peak_weight=0.0)
    flat = solve_epoch(state, slots, prices, cap_kw=3.0, dt_hours=1.0, penalty_unmet=25.0, peak_weight=10.0)
    peak_cheap = float(cheap.power_kw.sum(axis=0).max())
    peak_flat = float(flat.power_kw.sum(axis=0).max())
    assert peak_flat <= peak_cheap + 1e-6
    assert peak_flat <= 1.0 + 1e-5
