"""Plug congestion. Queue+EDF is the primary arm. Balk and mixed wait until later."""

from __future__ import annotations

from datetime import timedelta

from eci.optimize.mpc import ScenarioSession


def plug_occupancy(sessions: list[ScenarioSession], slot_minutes: int = 15) -> dict:
    """Time-weighted fraction of slots in which every plug is busy."""
    if not sessions:
        return {"binding_fraction": 0.0, "max_plugged": 0, "n_slots": 0}
    start = min(session.connection for session in sessions)
    end = max(session.disconnect for session in sessions)
    step = timedelta(minutes=slot_minutes)
    cursor = start.replace(second=0, microsecond=0)
    n_slots = 0
    binding = 0
    max_plugged = 0
    n_plugs = 54
    while cursor < end:
        plugged = sum(1 for session in sessions if session.connection <= cursor < session.disconnect)
        max_plugged = max(max_plugged, plugged)
        if plugged >= n_plugs:
            binding += 1
        n_slots += 1
        cursor += step
    return {
        "binding_fraction": (binding / n_slots) if n_slots else 0.0,
        "max_plugged": max_plugged,
        "n_slots": n_slots,
        "n_plugs": n_plugs,
        "arm": "queue_edf",
    }


def queue_edf(sessions: list[ScenarioSession], n_plugs: int = 54) -> list[ScenarioSession]:
    """Delay arrivals that would exceed ``n_plugs``, serving the earliest deadline first.

    Historical Caltech days rarely hit 54 simultaneous plugs. The arm still
    records the rule the later balk/mixed comparison will share.
    """
    remaining = sorted(sessions, key=lambda session: (session.connection, session.disconnect, session.session_id))
    active: list[ScenarioSession] = []
    accepted: list[ScenarioSession] = []
    waiting: list[ScenarioSession] = []
    for session in remaining:
        active = [item for item in active if item.disconnect > session.connection]
        waiting.append(session)
        waiting.sort(key=lambda item: (item.disconnect, item.connection, item.session_id))
        while waiting and len(active) < n_plugs:
            nxt = waiting.pop(0)
            start = max(nxt.connection, session.connection)
            moved = _shift(nxt, start)
            active.append(moved)
            accepted.append(moved)
    while waiting:
        if not active:
            accepted.append(waiting.pop(0))
            continue
        active.sort(key=lambda item: item.disconnect)
        freed = active.pop(0)
        nxt = waiting.pop(0)
        start = max(nxt.connection, freed.disconnect)
        moved = _shift(nxt, start)
        active.append(moved)
        accepted.append(moved)
    return accepted


def _shift(session: ScenarioSession, start) -> ScenarioSession:
    dwell = session.disconnect - session.connection
    return ScenarioSession(
        session_id=session.session_id,
        connection=start,
        disconnect=start + dwell,
        kwh=session.kwh,
        p_max_kw=session.p_max_kw,
        requested_departure=session.requested_departure,
        kwh_requested=session.kwh_requested,
        station_id=session.station_id,
    )
