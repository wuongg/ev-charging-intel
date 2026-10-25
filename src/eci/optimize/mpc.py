"""15-minute MPC. The first k slots of each solve are locked. T1 is not run."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from eci.optimize.feasibility import assert_feasible
from eci.optimize.formulation import slot_price
from eci.optimize.state import FleetState, Schedule, Vehicle
from eci.simulate.engine import clip_command, planning_end


@dataclass
class ScenarioSession:
    session_id: str
    connection: datetime
    disconnect: datetime
    kwh: float
    p_max_kw: float
    requested_departure: datetime | None = None
    kwh_requested: float | None = None
    station_id: str = ""


@dataclass
class ReplayLog:
    policy_name: str
    tier: str
    schedule: Schedule
    running_peak_kw: float
    dual_log: list[dict]
    epoch_costs: list[float] = field(default_factory=list)


def _floor_slot(moment: datetime, minutes: int) -> datetime:
    discard = (moment.minute % minutes) * 60 + moment.second
    return (moment - timedelta(seconds=discard)).replace(microsecond=0)


def _requirements(session: ScenarioSession, tier: str, median_dwell: timedelta) -> tuple[datetime, float]:
    if tier == "T3":
        end = session.disconnect
    elif tier == "T2":
        # T2v2 writes requested_departure before the replay. A missing one
        # falls back to the single median, which is the Day-4 imputation.
        end = session.requested_departure or (session.connection + median_dwell)
    else:
        raise ValueError(f"tier {tier} is not run on day 3")
    # kwh_requested is the energy the policy believes. Unmet is scored on session.kwh.
    energy = session.kwh_requested if session.kwh_requested is not None else session.kwh
    return end, float(energy)


class MPCLoop:
    def __init__(
        self,
        *,
        slot_minutes: int,
        committed_k: int,
        penalty_unmet: float,
        cap_kw: float,
        tariff_windows: list[dict],
        timezone_name: str,
        median_dwell: timedelta,
    ):
        self.slot_minutes = slot_minutes
        self.committed_k = committed_k
        self.penalty_unmet = penalty_unmet
        self.cap_kw = cap_kw
        self.tariff_windows = tariff_windows
        self.timezone_name = timezone_name
        self.median_dwell = median_dwell

    def run(self, sessions: list[ScenarioSession], policy, tier: str, policy_name: str, plant_hook=None) -> ReplayLog:
        if not sessions:
            raise ValueError("scenario day has no sessions")
        start = _floor_slot(min(session.connection for session in sessions), self.slot_minutes)
        end = max(session.disconnect for session in sessions)
        step = timedelta(minutes=self.slot_minutes)
        slots: list[datetime] = []
        cursor = start
        while cursor < end:
            slots.append(cursor)
            cursor += step
        n_i = len(sessions)
        n_t = len(slots)
        committed = np.zeros((n_i, n_t))
        delivered = np.zeros(n_i)
        dt = self.slot_minutes / 60.0
        requirements = [_requirements(session, tier, self.median_dwell) for session in sessions]
        prices_all = np.array(
            [slot_price(slot, self.tariff_windows, self.timezone_name) for slot in slots],
            dtype=float,
        )
        running_peak = 0.0
        dual_log: list[dict] = []
        epoch_costs: list[float] = []
        epoch = 0
        while epoch < n_t:
            now = slots[epoch]
            vehicles: list[Vehicle] = []
            active_index: list[int] = []
            for index, session in enumerate(sessions):
                belief_end, energy = requirements[index]
                if not (session.connection <= now < session.disconnect):
                    continue
                remaining = max(energy - delivered[index], 0.0)
                if remaining <= 1e-9:
                    continue
                vehicles.append(
                    Vehicle(
                        vehicle_id=session.session_id,
                        energy_remaining_kwh=remaining,
                        battery_max_kwh=max(energy, 0.0),
                        window_start=max(session.connection, now),
                        window_end=planning_end(belief_end, now, step),
                        p_max_kw=session.p_max_kw,
                        arrival=session.connection,
                    )
                )
                active_index.append(index)
            if not vehicles:
                epoch_costs.append(0.0)
                epoch += 1
                continue
            horizon_end = max(vehicle.window_end for vehicle in vehicles)
            horizon = [slot for slot in slots[epoch:] if slot < horizon_end]
            if not horizon:
                epoch_costs.append(0.0)
                epoch += 1
                continue
            prices = prices_all[epoch : epoch + len(horizon)]
            state = FleetState(vehicles=vehicles, running_peak_kw=running_peak, slot_minutes=self.slot_minutes)
            schedule = policy(state, horizon, prices, self.cap_kw, dt)
            assert_feasible(schedule, state, cap_kw=self.cap_kw)
            k = min(self.committed_k, len(horizon))
            for local, index in enumerate(active_index):
                session = sessions[index]
                commanded = clip_command(schedule.power_kw[local, :k], horizon[:k], session)
                committed[index, epoch : epoch + k] = commanded
                delivered[index] += float(commanded.sum() * dt)
            extra_cost = 0.0
            if plant_hook is not None:
                extra_cost = float(
                    plant_hook(epoch, active_index, delivered, requirements, committed) or 0.0
                )
            peak_now = float(committed[:, epoch : epoch + k].sum(axis=0).max()) if k else 0.0
            running_peak = max(running_peak, peak_now)
            slot_cost = float(np.sum(committed[:, epoch : epoch + k] * dt * prices_all[epoch : epoch + k]))
            epoch_costs.append(slot_cost + extra_cost)
            if schedule.cap_duals is not None:
                dual_log.append(
                    {
                        "epoch": now.isoformat(),
                        "cap_duals": [float(value) for value in np.asarray(schedule.cap_duals)[:k]],
                    }
                )
            epoch += k
        # Unmet is energy the car needed and did not receive before the physical unplug.
        unmet = np.array(
            [max(session.kwh - delivered[index], 0.0) for index, session in enumerate(sessions)],
            dtype=float,
        )
        # The committed trajectory is a plant record. Its feasible window is the
        # physical plug, which can outlast the deadline the policy believed.
        final_vehicles = []
        for index, session in enumerate(sessions):
            plan_energy = requirements[index][1]
            final_vehicles.append(
                Vehicle(
                    vehicle_id=session.session_id,
                    energy_remaining_kwh=session.kwh,
                    battery_max_kwh=max(session.kwh, plan_energy, 0.0),
                    window_start=session.connection,
                    window_end=session.disconnect,
                    p_max_kw=session.p_max_kw,
                    arrival=session.connection,
                )
            )
        final = Schedule(
            vehicle_ids=[session.session_id for session in sessions],
            slot_starts=slots,
            power_kw=committed,
            unmet_kwh=unmet,
            prices=prices_all,
            cap_kw=self.cap_kw,
            dt_hours=dt,
            status=policy_name,
            epoch_duals=dual_log,
        )
        assert_feasible(final, FleetState(final_vehicles, running_peak_kw=running_peak), cap_kw=self.cap_kw)
        from eci.eval.invariants import assert_invariants

        assert_invariants(final, sessions)
        return ReplayLog(
            policy_name=policy_name,
            tier=tier,
            schedule=final,
            running_peak_kw=running_peak,
            dual_log=dual_log,
            epoch_costs=epoch_costs,
        )
