"""I1 hard cap, I2 no phantom charge, I3 within p_max."""

from datetime import datetime, timezone

import numpy as np

from eci.eval.invariants import check_invariants
from eci.optimize.mpc import ScenarioSession
from eci.optimize.state import Schedule


def _session() -> ScenarioSession:
    return ScenarioSession(
        session_id="s",
        connection=datetime(2019, 1, 8, 0, tzinfo=timezone.utc),
        disconnect=datetime(2019, 1, 8, 2, tzinfo=timezone.utc),
        kwh=1.0,
        p_max_kw=1.0,
    )


def _schedule(power: np.ndarray, cap: float = 1.0) -> Schedule:
    start = datetime(2019, 1, 8, 0, tzinfo=timezone.utc)
    slots = [start.replace(hour=hour) for hour in range(power.shape[1])]
    return Schedule(
        vehicle_ids=["s"],
        slot_starts=slots,
        power_kw=power,
        unmet_kwh=np.zeros(1),
        prices=np.ones(power.shape[1]),
        cap_kw=cap,
        dt_hours=1.0,
        status="test",
    )


def test_i1_rejects_a_column_over_the_cap() -> None:
    violations = check_invariants(_schedule(np.array([[2.0, 0.0]]), cap=1.0), [_session()])
    assert any(item.startswith("I1") for item in violations)


def test_i2_rejects_power_before_arrival() -> None:
    session = _session()
    late = ScenarioSession(
        session_id="s",
        connection=datetime(2019, 1, 8, 1, tzinfo=timezone.utc),
        disconnect=session.disconnect,
        kwh=1.0,
        p_max_kw=1.0,
    )
    violations = check_invariants(_schedule(np.array([[1.0, 0.0]])), [late])
    assert any(item.startswith("I2") for item in violations)


def test_i3_rejects_power_above_p_max() -> None:
    violations = check_invariants(_schedule(np.array([[1.5, 0.0]]), cap=2.0), [_session()])
    assert any(item.startswith("I3") for item in violations)
