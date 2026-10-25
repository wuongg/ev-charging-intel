"""T2v2 draws the same deadline for the same session id."""

from datetime import datetime, timezone

import numpy as np
import polars as pl

from eci.optimize.impute import DwellSampler, sampler_from_frame, unit_interval
from eci.optimize.mpc import ScenarioSession


def _sampler() -> DwellSampler:
    hours = []
    for station in ("A", "B"):
        for hour in range(24):
            for dwell in (1.0, 2.0, 4.0, 8.0, 12.0):
                hours.append({"station_id": station, "hour": hour, "dwell_h": dwell + hour / 100.0})
    return sampler_from_frame(pl.DataFrame(hours))


def test_unit_interval_depends_only_on_the_id() -> None:
    assert unit_interval("s") == unit_interval("s")
    assert unit_interval("s") != unit_interval("other")
    assert 0.0 < unit_interval("s") < 1.0


def test_same_session_gets_the_same_deadline() -> None:
    sampler = _sampler()
    session = ScenarioSession(
        session_id="ev-7",
        connection=datetime(2019, 1, 8, 16, tzinfo=timezone.utc),
        disconnect=datetime(2019, 1, 8, 22, tzinfo=timezone.utc),
        kwh=8.0,
        p_max_kw=6.6,
        station_id="A",
    )
    first = sampler.apply([session])[0].requested_departure
    second = sampler.apply([session])[0].requested_departure
    assert first == second
    assert first is not None
    assert first > session.connection


def test_declared_departure_is_not_replaced() -> None:
    sampler = _sampler()
    declared = datetime(2019, 1, 8, 18, tzinfo=timezone.utc)
    session = ScenarioSession(
        session_id="ev-7",
        connection=datetime(2019, 1, 8, 16, tzinfo=timezone.utc),
        disconnect=datetime(2019, 1, 8, 23, tzinfo=timezone.utc),
        kwh=8.0,
        p_max_kw=6.6,
        requested_departure=declared,
        station_id="A",
    )
    assert sampler.apply([session])[0].requested_departure == declared


def test_same_tau_gives_the_same_deadline() -> None:
    sampler = _sampler()
    session = ScenarioSession(
        session_id="ev-7",
        connection=datetime(2019, 1, 8, 16, tzinfo=timezone.utc),
        disconnect=datetime(2019, 1, 8, 22, tzinfo=timezone.utc),
        kwh=8.0,
        p_max_kw=6.6,
        station_id="A",
    )
    first = sampler.apply_quantile([session], 0.75)[0].requested_departure
    second = sampler.apply_quantile([session], 0.75)[0].requested_departure
    low = sampler.apply_quantile([session], 0.25)[0].requested_departure
    assert first == second
    assert first is not None and low is not None
    assert low <= first


def test_quantile_is_inside_the_sample() -> None:
    sampler = _sampler()
    pool = sampler.pool("A", 8)
    draw = sampler.sample_hours("ev-7", "A", 8)
    assert float(pool.min()) - 1e-9 <= draw <= float(pool.max()) + 1e-9
    assert np.isfinite(draw)
