"""Flow vs HiGHS on small fleets. Skips when OR-Tools is not installed (CI default)."""

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from eci.config import REPO_ROOT
from eci.optimize.solve import solve_epoch
from eci.optimize.state import FleetState, Vehicle

ortools = pytest.importorskip("ortools")

from eci.optimize.flow import solve_epoch_flow  # noqa: E402

N_SCENARIOS = 20


def _fleet(seed: int) -> tuple[FleetState, list[datetime], np.ndarray, float]:
    rng = np.random.default_rng(seed)
    origin = datetime(2019, 3, 4, 16, tzinfo=timezone.utc)
    n_t = 8
    slots = [origin + timedelta(minutes=15 * t) for t in range(n_t)]
    prices = rng.uniform(0.08, 0.40, size=n_t)
    n_i = 6
    vehicles = []
    for index in range(n_i):
        start_slot = int(rng.integers(0, 4))
        length = int(rng.integers(3, 6))
        start = slots[start_slot]
        end = slots[min(start_slot + length, n_t - 1)] + timedelta(minutes=15)
        energy = float(rng.uniform(0.5, 2.5))
        vehicles.append(
            Vehicle(
                vehicle_id=f"s{seed}-ev{index}",
                energy_remaining_kwh=energy,
                battery_max_kwh=energy,
                window_start=start,
                window_end=end,
                p_max_kw=6.6,
                arrival=start,
            )
        )
    cap = float(rng.choice([8.0, 12.0, 20.0]))
    return FleetState(vehicles), slots, prices, cap


def test_flow_matches_highs_relative_gap() -> None:
    rows = []
    for seed in range(N_SCENARIOS):
        state, slots, prices, cap = _fleet(seed)
        started = time.perf_counter()
        lp = solve_epoch(state, slots, prices, cap, 0.25, 25.0)
        lp_s = time.perf_counter() - started
        started = time.perf_counter()
        flow = solve_epoch_flow(state, slots, prices, cap, 0.25, 25.0)
        flow_s = time.perf_counter() - started
        lp_cost = lp.costs(25.0)[0]
        flow_cost = flow.costs(25.0)[0]
        denom = max(abs(lp_cost), 1e-9)
        gap = abs(flow_cost - lp_cost) / denom
        assert gap <= 1e-6, f"seed={seed} gap={gap} lp={lp_cost} flow={flow_cost}"
        rows.append(
            {
                "scenario_id": seed,
                "lp_seconds": lp_s,
                "flow_seconds": flow_s,
                "lp_cost": lp_cost,
                "flow_cost": flow_cost,
                "rel_gap": gap,
            }
        )
    dest = Path(REPO_ROOT) / "results" / "flow_benchmark.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
