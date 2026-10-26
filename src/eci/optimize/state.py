"""Fleet state carried across MPC epochs.

Energy already delivered stays on the vehicle. ``running_peak_kw`` is the
metered peak so far. It is a floor for a later peak term and is not in today's
objective.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass
class Vehicle:
    vehicle_id: str
    energy_remaining_kwh: float
    battery_max_kwh: float
    window_start: datetime
    window_end: datetime
    p_max_kw: float
    arrival: datetime


@dataclass
class FleetState:
    vehicles: list[Vehicle]
    running_peak_kw: float = 0.0
    slot_minutes: int = 15


@dataclass
class Schedule:
    vehicle_ids: list[str]
    slot_starts: list[datetime]
    power_kw: np.ndarray
    unmet_kwh: np.ndarray
    prices: np.ndarray
    cap_kw: float
    dt_hours: float
    status: str
    cap_duals: np.ndarray | None = None
    peak_duals: np.ndarray | None = None
    epoch_duals: list[dict] = field(default_factory=list)

    def costs(self, penalty_unmet: float) -> tuple[float, float, float]:
        """Total, energy, and penalty cost. Prices are the tariff currency per kWh."""
        energy = float(np.sum(self.power_kw * self.dt_hours * np.asarray(self.prices, dtype=float)))
        penalty = float(penalty_unmet * np.sum(self.unmet_kwh))
        return energy + penalty, energy, penalty
