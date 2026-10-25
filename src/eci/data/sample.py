"""Synthetic sessions for the CI mini-pipeline. Not a scenario bank."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from eci.data.parse import format_rfc1123_utc


def make_sessions(n: int = 200, seed: int = 7) -> list[dict]:
    """``n`` Caltech-shaped sessions with a local arrival peak at 08:00 and ~5% clock skew."""
    rng = random.Random(seed)
    # 2019-01-07 16:00 UTC = 08:00 PST. January is standard time (UTC-8).
    origin = datetime(2019, 1, 7, 16, 0, 0, tzinfo=timezone.utc)
    sessions: list[dict] = []
    skew_every = 20  # 5%
    for index in range(n):
        day = index // 10
        minute = (index * 3) % 60
        connection = origin + timedelta(days=day, minutes=minute)
        disconnect = connection + timedelta(hours=5)
        done = connection + timedelta(hours=2)
        if index % skew_every == 0:
            done = disconnect + timedelta(minutes=3)
        if index == 1:
            disconnect = connection + timedelta(hours=50)
            done = connection + timedelta(hours=2)
        inputs = None
        if index == 2:
            inputs = [
                {
                    "WhPerMile": 250,
                    "kWhRequested": 8.0,
                    "milesRequested": 32,
                    "minutesAvailable": 120,
                    "modifiedAt": format_rfc1123_utc(connection - timedelta(minutes=2)),
                    "paymentRequired": True,
                    "requestedDeparture": format_rfc1123_utc(connection + timedelta(hours=4)),
                    "userID": 2,
                },
                {
                    "WhPerMile": 250,
                    "kWhRequested": 30.0,
                    "milesRequested": 120,
                    "minutesAvailable": 400,
                    "modifiedAt": format_rfc1123_utc(connection + timedelta(hours=2)),
                    "paymentRequired": True,
                    "requestedDeparture": format_rfc1123_utc(connection + timedelta(hours=6)),
                    "userID": 2,
                },
            ]
        sessions.append(
            {
                "_id": f"mini-{index}",
                "clusterID": "0039",
                "connectionTime": format_rfc1123_utc(connection),
                "disconnectTime": format_rfc1123_utc(disconnect),
                "doneChargingTime": format_rfc1123_utc(done),
                "kWhDelivered": round(rng.uniform(1.0, 20.0), 3),
                "sessionID": f"mini-session-{index}",
                "siteID": "0002",
                "spaceID": f"CA-{index}",
                "stationID": f"2-39-78-{index}",
                "timezone": "America/Los_Angeles",
                "userID": None if index % 11 == 0 else f"user-{index}",
                "userInputs": inputs,
            }
        )
    return sessions
