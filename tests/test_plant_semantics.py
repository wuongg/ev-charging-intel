"""The plant delivers until the recorded disconnect. The policy plans on its belief."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from eci.eval.invariants import check_invariants
from eci.optimize.mpc import MPCLoop, ScenarioSession
from eci.optimize.policies import charge_on_arrival
from eci.optimize.state import Schedule


def _windows() -> list[dict]:
    return [
        {
            "days": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
            "start": "00:00",
            "end": "24:00",
            "price_per_kwh": 0.10,
        }
    ]


def test_plant_charges_past_an_early_declared_deadline() -> None:
    start = datetime(2019, 1, 8, 16, 0, tzinfo=timezone.utc)
    declared = start + timedelta(hours=1)
    disconnect = start + timedelta(hours=4)
    session = ScenarioSession(
        session_id="only",
        connection=start,
        disconnect=disconnect,
        kwh=10.0,
        p_max_kw=6.0,
        requested_departure=declared,
    )
    seen: list[datetime] = []

    def policy(state, slots, prices, cap, dt):
        seen.append(state.vehicles[0].window_end)
        return charge_on_arrival(state, slots, prices, cap, dt)

    loop = MPCLoop(
        slot_minutes=15,
        committed_k=1,
        penalty_unmet=25.0,
        cap_kw=50.0,
        tariff_windows=_windows(),
        timezone_name="UTC",
        median_dwell=timedelta(hours=5),
    )
    log = loop.run([session], policy, "T2", "P0")
    delivered = float(log.schedule.power_kw.sum() * log.schedule.dt_hours)
    after_deadline = [
        slot
        for slot, power in zip(log.schedule.slot_starts, log.schedule.power_kw[0], strict=True)
        if power > 1e-6 and slot >= declared
    ]
    assert after_deadline, "plant stopped at the declared deadline"
    assert all(slot < disconnect for slot in after_deadline)
    assert delivered == pytest.approx(10.0)
    assert float(log.schedule.unmet_kwh[0]) == pytest.approx(0.0)
    assert seen[0] == declared
    assert seen[0] != disconnect
    assert check_invariants(log.schedule, [session]) == []


def test_i9_allows_power_after_the_declared_deadline() -> None:
    start = datetime(2019, 1, 8, 16, 0, tzinfo=timezone.utc)
    declared = start + timedelta(hours=1)
    disconnect = start + timedelta(hours=2)
    session = ScenarioSession(
        session_id="only",
        connection=start,
        disconnect=disconnect,
        kwh=6.0,
        p_max_kw=6.0,
        requested_departure=declared,
    )
    slots = [start, start + timedelta(minutes=15), start + timedelta(hours=1)]
    schedule = Schedule(
        vehicle_ids=["only"],
        slot_starts=slots,
        power_kw=np.array([[6.0, 6.0, 6.0]]),
        unmet_kwh=np.array([0.0]),
        prices=np.array([0.1, 0.1, 0.1]),
        cap_kw=50.0,
        dt_hours=0.25,
        status="hand",
    )
    assert "I9 plant" not in check_invariants(schedule, [session])
