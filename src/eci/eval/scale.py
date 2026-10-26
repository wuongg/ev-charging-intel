"""Labeled scenario knobs: scale dwell, tariff spread, and utilization."""

from __future__ import annotations

from dataclasses import replace

from eci.eval.dispatch import _normalize_session_id, _source_key
from eci.optimize.impute import unit_interval
from eci.optimize.mpc import ScenarioSession
from eci.simulate.congestion import queue_edf

N_PLUGS = 54
SCALE_NOTE = (
    "Scenario analysis: actual disconnect and believed deadline both scaled by dwell_scale; "
    "tariff_spread stretches peak minus off-peak and keeps the off-peak price. Not a measurement."
)


def scale_dwell(sessions: list[ScenarioSession], factor: float) -> list[ScenarioSession]:
    """Stretch stay and declared window by the same factor. Energy is unchanged."""
    if abs(factor - 1.0) < 1e-12:
        return list(sessions)
    if factor <= 0:
        raise ValueError(f"dwell scale must be positive, got {factor}")
    out = []
    for session in sessions:
        stay = session.disconnect - session.connection
        requested = None
        if session.requested_departure is not None:
            requested = session.connection + (session.requested_departure - session.connection) * factor
        out.append(
            replace(
                session,
                disconnect=session.connection + stay * factor,
                requested_departure=requested,
            )
        )
    return out


def scale_spread(windows: list[dict], factor: float) -> list[dict]:
    """Keep the cheapest window. Stretch every other price away from it by ``factor``."""
    if factor <= 0:
        raise ValueError(f"spread scale must be positive, got {factor}")
    off = min(float(window["price_per_kwh"]) for window in windows)
    out = []
    for window in windows:
        row = dict(window)
        price = float(window["price_per_kwh"])
        row["price_per_kwh"] = off + factor * (price - off)
        out.append(row)
    return out


def apply_util(sessions: list[ScenarioSession], factor: float) -> list[ScenarioSession]:
    """factor < 1 drops sessions; factor > 1 copies then queues on 54 plugs."""
    if factor <= 0:
        raise ValueError(f"utilization must be positive, got {factor}")
    if abs(factor - 1.0) < 1e-12:
        return list(sessions)
    if factor < 1.0:
        kept = [
            session
            for session in sessions
            if unit_interval(_normalize_session_id(_source_key(session.session_id))) < factor
        ]
        return kept or list(sessions[:1])
    copies: list[ScenarioSession] = []
    whole = int(factor)
    frac = factor - whole
    for copy in range(whole):
        for session in sessions:
            if copy == 0:
                copies.append(session)
                continue
            copies.append(
                replace(
                    session,
                    session_id=f"{session.session_id}#x{copy}",
                    requested_departure=None,
                    kwh_requested=None,
                )
            )
    if frac > 1e-12:
        for session in apply_util(sessions, frac):
            copies.append(replace(session, session_id=f"{session.session_id}#frac"))
    return queue_edf(copies, n_plugs=N_PLUGS)
