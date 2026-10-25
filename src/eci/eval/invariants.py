"""Hard checks on a committed schedule. Solver status is not evidence."""

from __future__ import annotations

import numpy as np

from eci.optimize.mpc import ScenarioSession
from eci.optimize.state import Schedule


class InvariantError(RuntimeError):
    def __init__(self, violations: list[str]):
        self.violations = violations
        super().__init__("; ".join(violations))


def check_invariants(
    schedule: Schedule,
    sessions: list[ScenarioSession],
    *,
    tol: float = 1e-4,
) -> list[str]:
    """I1 hard cap, I2 no energy outside the physical plug, I3 command within p_max."""
    violations: list[str] = []
    power = np.asarray(schedule.power_kw, dtype=float)
    if power.ndim != 2 or power.shape[0] != len(sessions):
        return [f"I1 power rows {getattr(power, 'shape', None)} != {len(sessions)} sessions"]
    column = power.sum(axis=0)
    if np.any(column > schedule.cap_kw + tol):
        violations.append("I1 cap")
    if np.any(power < -tol):
        violations.append("I1 negative power")
    for index, session in enumerate(sessions):
        for t, slot in enumerate(schedule.slot_starts):
            if power[index, t] <= tol:
                continue
            if not (session.connection <= slot < session.disconnect):
                violations.append("I2 phantom")
                break
            if power[index, t] > session.p_max_kw + tol:
                violations.append("I3 c-rate")
                break
    return violations


def assert_invariants(schedule: Schedule, sessions: list[ScenarioSession], *, tol: float = 1e-4) -> None:
    violations = check_invariants(schedule, sessions, tol=tol)
    if violations:
        raise InvariantError(violations)
