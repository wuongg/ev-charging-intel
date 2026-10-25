"""Hazard metrics on synthetic sessions. The real mirror is scored by the train script."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import polars as pl

from eci.data.measures import early_unplug_compliance, iced_proxy_session_level, overdispersion_test, overstay_rate
from eci.data.parse import format_rfc1123_utc

UTC = timezone.utc
LA = ZoneInfo("America/Los_Angeles")


def test_early_unplug_splits_benign_and_violating() -> None:
    requested = datetime(2018, 6, 1, 16, tzinfo=UTC)
    snapshots = [
        {"requestedDeparture": format_rfc1123_utc(requested), "kWhRequested": 10},
        {"requestedDeparture": format_rfc1123_utc(requested), "kWhRequested": 10},
        {"requestedDeparture": format_rfc1123_utc(requested), "kWhRequested": 10},
    ]
    frame = pl.DataFrame(
        {
            "disconnectTime_utc": [
                datetime(2018, 6, 1, 14, tzinfo=UTC),
                datetime(2018, 6, 1, 14, tzinfo=UTC),
                datetime(2018, 6, 1, 16, tzinfo=UTC),
            ],
            "kWhDelivered": [10.0, 1.0, 10.0],
            "userInputs_last": [json.dumps(item) for item in snapshots],
        }
    )
    result = early_unplug_compliance(frame)
    assert result["n"] == 3
    assert result["p_early"] == 2 / 3
    assert result["p_benign"] == 1 / 3
    assert result["p_violating"] == 1 / 3
    assert result["low_confidence"] is True
    assert result["median_gap_min"] > 0


def test_overstay_drops_clock_skew() -> None:
    base = datetime(2018, 6, 2, 12, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "doneChargingTime_utc": [base, base, base, base + timedelta(hours=3)],
            "disconnectTime_utc": [
                base + timedelta(minutes=20),
                base + timedelta(minutes=90),
                base + timedelta(minutes=5),
                base,
            ],
            "f_clock_skew": [False, False, False, True],
        }
    )
    result = overstay_rate(frame)
    assert result["n"] == 3
    assert result["p_over_15min"] == 2 / 3
    assert result["p_over_1h"] == 1 / 3


def test_iced_proxy_flags_low_energy() -> None:
    start = datetime(2018, 6, 3, 8, tzinfo=UTC)
    dwells = [1.0] * 6 + [4.0]
    energy = [6.0] * 6 + [1.0]
    frame = pl.DataFrame(
        {
            "stationID": ["S"] * 7,
            "connectionTime_utc": [start] * 7,
            "disconnectTime_utc": [start + timedelta(hours=dwell) for dwell in dwells],
            "kWhDelivered": energy,
        }
    )
    result = iced_proxy_session_level(frame)
    assert result["n"] == 7
    assert result["n_iced"] == 1
    assert result["low_confidence"] is True


def test_overdispersion_flags_bursty_hour() -> None:
    start = datetime(2018, 6, 4, tzinfo=LA)
    stamps: list[datetime] = []
    counts: list[int] = []
    for day in range(30):
        stamps.append(start + timedelta(days=day))
        counts.append(2)
        stamps.append(start + timedelta(days=day, hours=1))
        counts.append(0 if day % 2 == 0 else 20)
    hourly = pl.DataFrame({"ts_local_hour": stamps, "n_sessions": counts})
    result = overdispersion_test(hourly)
    assert result["poisson_inadequate"] is True
    assert result["mean_ratio"] is not None and result["mean_ratio"] > 2
    assert result["low_confidence"] is True
