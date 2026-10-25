"""Golden parse of a normal session, a DST crossing, and a multi-input snapshot."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from eci.data.clean import clean_sessions
from eci.data.features import build_hourly_demand
from eci.data.ingest import recover_sessions
from eci.data.parse import format_rfc1123_utc, parse_sessions
from eci.data.schema_contract import SESSION_KEYS, coerce_session
from eci.forecast.naive import seasonal_naive


def _session(**overrides) -> dict:
    base = {
        "_id": "norm",
        "clusterID": "0039",
        "connectionTime": "Wed, 25 Apr 2018 11:08:04 GMT",
        "disconnectTime": "Wed, 25 Apr 2018 13:20:10 GMT",
        "doneChargingTime": "Wed, 25 Apr 2018 12:30:00 GMT",
        "kWhDelivered": 7.932,
        "sessionID": "s-normal",
        "siteID": "0002",
        "spaceID": "CA-496",
        "stationID": "2-39-78-362",
        "timezone": "America/Los_Angeles",
        "userID": None,
        "userInputs": [
            {
                "WhPerMile": 250,
                "kWhRequested": 10.0,
                "milesRequested": 40,
                "minutesAvailable": 120,
                "modifiedAt": "Wed, 25 Apr 2018 11:08:04 GMT",
                "paymentRequired": True,
                "requestedDeparture": "Wed, 25 Apr 2018 13:08:04 GMT",
                "userID": 1,
            }
        ],
    }
    base.update(overrides)
    return base


def test_session_schema_has_13_keys() -> None:
    assert len(SESSION_KEYS) == 13


def test_site_id_zero_pad() -> None:
    coerced, changed = coerce_session(_session(siteID=2))
    assert coerced["siteID"] == "0002"
    assert changed is True
    untouched, same = coerce_session(_session(siteID="0002"))
    assert untouched["siteID"] == "0002"
    assert same is False


def test_golden_normal_utc_and_local() -> None:
    frame = parse_sessions([_session()])
    row = frame.to_dicts()[0]
    utc = row["connectionTime_utc"]
    local = row["connectionTime_local"]
    assert utc.hour == 11 and utc.minute == 8 and utc.second == 4
    assert utc.utcoffset() == timedelta(0)
    # 2018-04-25 is PDT (UTC-7): 11:08 UTC is 04:08 local.
    assert local.hour == 4 and local.minute == 8
    assert local.utcoffset() == timedelta(hours=-7)
    snapshot = json.loads(row["userInputs_last"])
    assert snapshot["kWhRequested"] == 10.0


def test_golden_dst_crossing() -> None:
    # Spring forward 2019-03-10. 18:00 PST -> 10:00 PDT is 15h UTC and 16h on the wall clock.
    frame = parse_sessions(
        [
            _session(
                _id="dst",
                connectionTime="Sun, 10 Mar 2019 02:00:00 GMT",
                disconnectTime="Sun, 10 Mar 2019 17:00:00 GMT",
                doneChargingTime="Sun, 10 Mar 2019 12:00:00 GMT",
            )
        ]
    )
    row = frame.to_dicts()[0]
    assert row["connectionTime_local"].day == 9
    assert row["connectionTime_local"].hour == 18
    assert row["connectionTime_local"].utcoffset() == timedelta(hours=-8)
    assert row["disconnectTime_local"].day == 10
    assert row["disconnectTime_local"].hour == 10
    assert row["disconnectTime_local"].utcoffset() == timedelta(hours=-7)
    cleaned, _funnel = clean_sessions(frame)
    assert cleaned["f_dst_cross"][0] is True


def test_golden_multi_input_snapshot_drops_post_hoc() -> None:
    connection = "Wed, 25 Apr 2018 11:08:04 GMT"
    frame = parse_sessions(
        [
            _session(
                _id="multi",
                connectionTime=connection,
                userInputs=[
                    {
                        "kWhRequested": 10.0,
                        "minutesAvailable": 60,
                        "WhPerMile": 250,
                        "modifiedAt": "Wed, 25 Apr 2018 11:00:00 GMT",
                    },
                    {
                        "kWhRequested": 20.0,
                        "minutesAvailable": 90,
                        "WhPerMile": 280,
                        "modifiedAt": "Wed, 25 Apr 2018 11:07:00 GMT",
                    },
                    {
                        "kWhRequested": 99.0,
                        "minutesAvailable": 30,
                        "WhPerMile": 250,
                        "modifiedAt": "Wed, 25 Apr 2018 12:00:00 GMT",
                    },
                ],
            )
        ]
    )
    snapshot = json.loads(frame["userInputs_last"][0])
    assert snapshot["kWhRequested"] == 20.0


def test_acn_modified_at_after_plugin_uses_earliest() -> None:
    """The mirror stamps modifiedAt seconds after connectionTime and has no createdOn."""
    frame = parse_sessions(
        [
            _session(
                connectionTime="Mon, 07 May 2018 14:38:18 GMT",
                disconnectTime="Tue, 08 May 2018 01:15:01 GMT",
                doneChargingTime="Tue, 08 May 2018 01:14:53 GMT",
                userInputs=[
                    {
                        "kWhRequested": 8.0,
                        "minutesAvailable": 60,
                        "WhPerMile": 400,
                        "modifiedAt": "Mon, 07 May 2018 14:38:32 GMT",
                    },
                    {
                        "kWhRequested": 28.0,
                        "minutesAvailable": 648,
                        "WhPerMile": 400,
                        "modifiedAt": "Mon, 07 May 2018 15:41:04 GMT",
                    },
                ],
            )
        ]
    )
    snapshot = json.loads(frame["userInputs_last"][0])
    assert snapshot["kWhRequested"] == 8.0


def test_created_on_beats_modified_at() -> None:
    frame = parse_sessions(
        [
            _session(
                userInputs=[
                    {
                        "kWhRequested": 4.0,
                        "minutesAvailable": 30,
                        "WhPerMile": 200,
                        "modifiedAt": "Wed, 25 Apr 2018 11:08:04 GMT",
                        "createdOn": "Wed, 25 Apr 2018 10:00:00 GMT",
                    },
                    {
                        "kWhRequested": 6.0,
                        "minutesAvailable": 40,
                        "WhPerMile": 200,
                        "createdOn": "Wed, 25 Apr 2018 11:05:00 GMT",
                    },
                ]
            )
        ]
    )
    snapshot = json.loads(frame["userInputs_last"][0])
    assert snapshot["kWhRequested"] == 6.0


def test_tiny_sessions_flags(tiny_sessions: list[dict]) -> None:
    parsed = parse_sessions(tiny_sessions)
    cleaned, funnel = clean_sessions(parsed)
    assert cleaned.height == 10
    assert cleaned.filter(pl.col("f_clock_skew")).height == 1
    assert cleaned.filter(pl.col("f_long_dwell")).height == 1
    assert cleaned.filter(pl.col("f_ui_absurd")).height == 1
    multi = cleaned.filter(pl.col("_id") == "tiny-2").to_dicts()[0]
    assert json.loads(multi["userInputs_last"])["kWhRequested"] == 15.0
    absurd = cleaned.filter(pl.col("_id") == "tiny-3").to_dicts()[0]
    assert absurd["userInputs_last"] is None
    assert funnel.filter(pl.col("stage") == "V3")["n_excluded"][0] == 0


def test_time_order_and_kwh_flag() -> None:
    bad_order = _session(
        _id="backwards",
        connectionTime="Wed, 25 Apr 2018 13:00:00 GMT",
        disconnectTime="Wed, 25 Apr 2018 11:00:00 GMT",
        doneChargingTime="Wed, 25 Apr 2018 12:00:00 GMT",
    )
    huge = _session(_id="huge", kWhDelivered=150.0)
    parsed = parse_sessions([bad_order, huge])
    cleaned, funnel = clean_sessions(parsed)
    assert cleaned.height == 1
    assert cleaned["f_kwh_oor"][0] is True
    assert funnel.filter(pl.col("stage") == "V3")["n_excluded"][0] == 1


def test_duplicate_keeps_first() -> None:
    first = _session(_id="a", kWhDelivered=1.0)
    second = _session(_id="b", kWhDelivered=2.0)
    cleaned, funnel = clean_sessions(parse_sessions([first, second]))
    assert cleaned.height == 1
    assert cleaned["kWhDelivered"][0] == 1.0
    assert funnel.filter(pl.col("stage") == "R3")["n_excluded"][0] == 1


def test_truncated_json_recovers_complete_objects() -> None:
    text = (
        '{"_meta": {"site": "caltech"}, "_items": '
        '[{"sessionID": "a", "connectionTime": "Wed, 25 Apr 2018 11:08:04 GMT"}, '
        '{"sessionID": "b", "kWhDelivered": 1},'
    )
    sessions, truncated = recover_sessions(text, byte_length=64)
    assert truncated is True
    assert [row["sessionID"] for row in sessions] == ["a", "b"]


def test_hourly_demand_uses_connection_time_and_censors() -> None:
    connection = datetime(2019, 1, 15, 16, 0, tzinfo=timezone.utc)  # 08:00 PST
    done = connection + timedelta(hours=7)  # 15:00 local — must not receive the kWh
    disconnect = connection + timedelta(hours=8)
    record = _session(
        connectionTime=format_rfc1123_utc(connection),
        disconnectTime=format_rfc1123_utc(disconnect),
        doneChargingTime=format_rfc1123_utc(done),
        kWhDelivered=4.0,
        userInputs=None,
    )
    cleaned, _funnel = clean_sessions(parse_sessions([record]))
    hourly = build_hourly_demand(cleaned, "caltech")
    by_hour = {row["ts_local_hour"].hour: row["kWh_delivered"] for row in hourly.to_dicts()}
    assert by_hour[8] == pytest.approx(4.0)
    assert by_hour.get(15, 0.0) == pytest.approx(0.0)

    censored = build_hourly_demand(cleaned, "caltech", window_end=connection + timedelta(hours=1))
    assert censored.height == 0


def test_seasonal_naive_lags_one_week() -> None:
    start = datetime(2019, 1, 1, tzinfo=timezone.utc)
    hours = pl.datetime_range(
        start,
        start + timedelta(hours=200),
        interval="1h",
        time_zone="UTC",
        eager=True,
    )
    values = list(range(len(hours)))
    series = pl.DataFrame({"ts_local_hour": hours, "kWh_delivered": values})
    forecast = seasonal_naive(series, m=168)
    assert forecast["yhat"][168] == 0
    assert forecast["yhat"][169] == 1
    assert forecast["q50"][168] is not None
