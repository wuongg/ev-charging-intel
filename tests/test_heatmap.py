"""Dwell/spread knobs and monotonicity flags. Bank runs live in scripts."""

from datetime import datetime, timedelta, timezone

import polars as pl

from eci.eval.heatmap import break_even_hours, dwell_tariff_map, monotonicity_flags
from eci.eval.scale import scale_dwell, scale_spread
from eci.optimize.mpc import ScenarioSession


def _session() -> ScenarioSession:
    connection = datetime(2018, 6, 4, 16, tzinfo=timezone.utc)
    return ScenarioSession(
        session_id="src-1",
        connection=connection,
        disconnect=connection + timedelta(hours=4),
        kwh=8.0,
        p_max_kw=6.6,
        requested_departure=connection + timedelta(hours=3),
    )


def test_scale_dwell_moves_plant_and_belief() -> None:
    session = scale_dwell([_session()], 2.0)[0]
    assert (session.disconnect - session.connection) == timedelta(hours=8)
    assert session.requested_departure is not None
    assert (session.requested_departure - session.connection) == timedelta(hours=6)


def test_scale_spread_keeps_off_peak() -> None:
    windows = [
        {"name": "off", "price_per_kwh": 0.05, "days": ["mon"], "start": "00:00", "end": "08:00"},
        {"name": "on", "price_per_kwh": 0.25, "days": ["mon"], "start": "12:00", "end": "18:00"},
    ]
    scaled = scale_spread(windows, 2.0)
    assert scaled[0]["price_per_kwh"] == 0.05
    assert abs(scaled[1]["price_per_kwh"] - 0.45) < 1e-12


def test_monotonicity_flags_a_drop_and_break_even_reads_the_first_cell() -> None:
    rows = []
    for scale in (0.25, 0.5, 1.0, 2.0):
        rows.append(
            {
                "dwell_scale": scale,
                "tariff_spread": 1.0,
                "policy": "P3",
                "reduction_vs_p0": 0.01 if scale < 2 else 0.05,
                "energy_reduction_vs_p0": 0.01 if scale < 2 else 0.05,
                "voi_p3_minus_p4": 1.0,
                "unmet_rate": 0.1,
            }
        )
    rows.append(
        {
            "dwell_scale": 4.0,
            "tariff_spread": 1.0,
            "policy": "P3",
            "reduction_vs_p0": 0.04,
            "energy_reduction_vs_p0": 0.04,
            "voi_p3_minus_p4": 1.0,
            "unmet_rate": 0.1,
        }
    )
    summary = dwell_tariff_map(pl.DataFrame(rows), median_hours=5.0)
    flags = monotonicity_flags(summary)
    assert flags
    even = break_even_hours(summary, 5.0)
    assert even["found"]
    assert even["dwell_scale"] == 2.0
