"""S1 outage is seeded; I1 still holds when the plant zeros outage epochs."""

from datetime import timedelta

from eci.eval.chaos import synthetic_week
from eci.eval.invariants import check_invariants
from eci.optimize.mpc import MPCLoop
from eci.optimize.solve import solve_epoch_fast
from eci.simulate.vn_scenarios import (
    Outage,
    compare_outage,
    load_s1,
    outage_hook_factory,
    sample_outages,
)


def test_outage_injection_is_deterministic() -> None:
    params = load_s1()
    sessions = synthetic_week(n_days=14, per_day=6)
    days = sorted({session.connection.date() for session in sessions})
    first = sample_outages(days, params, seed=9, timezone_name="UTC")
    second = sample_outages(days, params, seed=9, timezone_name="UTC")
    assert [(event.start, event.end) for event in first] == [(event.start, event.end) for event in second]


def test_i1_passes_with_outage() -> None:
    sessions = synthetic_week(n_days=3, per_day=6)
    start = min(session.connection for session in sessions)
    slots = [start + timedelta(minutes=15 * step) for step in range(4 * 12)]
    first = min(session.connection for session in sessions)
    outages = [Outage(start=first, end=first + timedelta(hours=2))]
    hook = outage_hook_factory(slots, outages, 0.25)
    loop = MPCLoop(
        slot_minutes=15,
        committed_k=1,
        penalty_unmet=25.0,
        cap_kw=40.0,
        tariff_windows=[
            {
                "name": "flat",
                "start": "00:00",
                "end": "24:00",
                "days": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
                "price_per_kwh": 0.10,
            }
        ],
        timezone_name="UTC",
        median_dwell=timedelta(hours=4),
    )

    def policy(state, slot_starts, prices, cap, dt):
        return solve_epoch_fast(state, slot_starts, prices, cap, dt, 25.0)

    log = loop.run(sessions, policy, "T3", "S1", plant_hook=hook)
    violations = check_invariants(log.schedule, sessions)
    assert not any(item.startswith("I1") for item in violations)


def test_compare_outage_keeps_i1() -> None:
    sessions = synthetic_week(n_days=2, per_day=4)
    params = load_s1()
    days = sorted({session.connection.date() for session in sessions})
    outages = sample_outages(days, params, seed=9, timezone_name="America/Los_Angeles")
    windows = [
        {
            "name": "flat",
            "start": "00:00",
            "end": "24:00",
            "days": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
            "price_per_kwh": 0.10,
        }
    ]
    rows = compare_outage(
        sessions,
        cap_kw=40.0,
        windows=windows,
        median_dwell=timedelta(hours=4),
        penalty=25.0,
        demand_rate=0.0,
        outages=outages,
    )
    assert all(row["i1_ok"] for row in rows)
    names = {row["scenario"] for row in rows}
    assert names == {"P0_outage_off", "P0_outage_on", "P3_outage_off", "P3_outage_on"}
