"""The checker rejects cap, window, and energy violations without trusting the solver."""

from datetime import datetime, timezone

import numpy as np

from eci.optimize.feasibility import check_feasibility
from eci.optimize.state import FleetState, Schedule, Vehicle


def _state() -> FleetState:
    start = datetime(2019, 6, 3, 16, tzinfo=timezone.utc)
    end = datetime(2019, 6, 3, 20, tzinfo=timezone.utc)
    return FleetState(
        [
            Vehicle(
                vehicle_id="ev",
                energy_remaining_kwh=2.0,
                battery_max_kwh=2.0,
                window_start=start,
                window_end=end,
                p_max_kw=1.0,
                arrival=start,
            )
        ]
    )


def _schedule(power: np.ndarray, unmet: float) -> Schedule:
    start = datetime(2019, 6, 3, 16, tzinfo=timezone.utc)
    slots = [start.replace(hour=16 + hour) for hour in range(4)]
    return Schedule(
        vehicle_ids=["ev"],
        slot_starts=slots,
        power_kw=power,
        unmet_kwh=np.array([unmet]),
        prices=np.ones(4),
        cap_kw=1.0,
        dt_hours=1.0,
        status="test",
    )


def test_valid_schedule_passes() -> None:
    power = np.array([[1.0, 1.0, 0.0, 0.0]])
    assert check_feasibility(_schedule(power, 0.0), _state()) == []


def test_cap_violation_is_caught() -> None:
    power = np.array([[2.0, 0.0, 0.0, 0.0]])
    violations = check_feasibility(_schedule(power, 0.0), _state(), cap_kw=1.0)
    assert any("cap" in item or "p_max" in item for item in violations)


def test_energy_shortfall_is_caught() -> None:
    power = np.array([[1.0, 0.0, 0.0, 0.0]])
    violations = check_feasibility(_schedule(power, 0.0), _state())
    assert any("energy" in item for item in violations)


def test_power_outside_window_is_caught() -> None:
    start = datetime(2019, 6, 3, 15, tzinfo=timezone.utc)
    slots = [start.replace(hour=15 + hour) for hour in range(4)]
    schedule = _schedule(np.array([[1.0, 1.0, 0.0, 0.0]]), 0.0)
    schedule.slot_starts = slots
    violations = check_feasibility(schedule, _state())
    assert any("window" in item for item in violations)
