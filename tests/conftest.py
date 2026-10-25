"""Ten synthetic sessions: clock skew, long dwell, multi-input, absurd input."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from eci.data.parse import format_rfc1123_utc


@pytest.fixture
def tiny_sessions() -> list[dict]:
    base = datetime(2019, 1, 15, 16, 0, 0, tzinfo=timezone.utc)
    rows: list[dict] = []
    for index in range(10):
        connection = base + timedelta(hours=index)
        disconnect = connection + timedelta(hours=4)
        done = connection + timedelta(hours=2)
        inputs = None
        user_id = None if index == 4 else f"user-{index}"
        if index == 0:
            done = disconnect + timedelta(minutes=5)
        elif index == 1:
            disconnect = connection + timedelta(hours=50)
            done = connection + timedelta(hours=2)
        elif index == 2:
            inputs = [
                _input(10.0, 60, 250, connection - timedelta(minutes=10)),
                _input(15.0, 90, 250, connection - timedelta(minutes=1)),
                _input(40.0, 400, 250, connection + timedelta(hours=3)),
            ]
        elif index == 3:
            inputs = [_input(0.0, 1, 900, connection - timedelta(minutes=1))]
        rows.append(
            {
                "_id": f"tiny-{index}",
                "clusterID": "0039",
                "connectionTime": format_rfc1123_utc(connection),
                "disconnectTime": format_rfc1123_utc(disconnect),
                "doneChargingTime": format_rfc1123_utc(done),
                "kWhDelivered": 5.0 + index,
                "sessionID": f"tiny-session-{index}",
                "siteID": "0002",
                "spaceID": f"CA-{index}",
                "stationID": f"2-39-78-{index}",
                "timezone": "America/Los_Angeles",
                "userID": user_id,
                "userInputs": inputs,
            }
        )
    return rows


def _input(kwh: float, minutes: int, wh: int, moment: datetime) -> dict:
    return {
        "WhPerMile": wh,
        "kWhRequested": kwh,
        "milesRequested": 10,
        "minutesAvailable": minutes,
        "modifiedAt": format_rfc1123_utc(moment),
        "paymentRequired": True,
        "requestedDeparture": format_rfc1123_utc(moment + timedelta(hours=2)),
        "userID": 1,
    }
