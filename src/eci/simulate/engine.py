"""Physical plant. The declared deadline never enters this module.

A car is plugged from its recorded connection to its recorded disconnect.
Policy code may plan on a shorter or longer belief. Only this module decides
whether a commanded kilowatt is delivered.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np


def admits(session, moment: datetime) -> bool:
    """True when the car is physically plugged at the start of ``moment``."""
    return session.connection <= moment < session.disconnect


def clip_command(power: np.ndarray, slots: list[datetime], session) -> np.ndarray:
    """Zero every commanded slot that falls outside the physical plug."""
    out = np.array(power, dtype=float, copy=True)
    for index, slot in enumerate(slots):
        if not admits(session, slot):
            out[index] = 0.0
    return out


def planning_end(belief_end: datetime, now: datetime, step: timedelta) -> datetime:
    """Horizon the policy may see.

    A belief still in the future is used as-is, including one that runs past
    the true disconnect. That over-long plan is how a late declaration wastes
    energy: the plant will refuse the slots after the car leaves.

    A belief that has already expired does not pretend the car left. The plug
    is observable. The policy is then told only the current step, not the
    future disconnect.
    """
    if belief_end > now:
        return belief_end
    return now + step
