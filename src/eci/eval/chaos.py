"""CH-01, CH-06, and CH-08 on one common-random-number week."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from eci.eval.invariants import check_invariants
from eci.optimize.mpc import MPCLoop, ScenarioSession
from eci.optimize.solve import solve_epoch_fast


def drop_sessions(sessions: list[ScenarioSession], frac: float, seed: int = 7) -> list[ScenarioSession]:
    if not sessions or frac <= 0:
        return list(sessions)
    import numpy as np

    rng = np.random.default_rng(seed)
    keep = rng.random(len(sessions)) >= frac
    if not keep.any():
        keep[0] = True
    return [session for session, flag in zip(sessions, keep, strict=True) if flag]


def force_infeasible(cap_kw: float, cut: float = 0.5) -> float:
    return cap_kw * cut


def synthetic_week(n_days: int = 7, per_day: int = 8) -> list[ScenarioSession]:
    """One workplace week. Deterministic, no parquet."""
    sessions: list[ScenarioSession] = []
    origin = datetime(2019, 1, 7, 16, tzinfo=timezone.utc)  # 08:00 Los Angeles
    for day in range(n_days):
        for index in range(per_day):
            connection = origin + timedelta(days=day, minutes=15 * index)
            sessions.append(
                ScenarioSession(
                    session_id=f"d{day}-s{index}",
                    connection=connection,
                    disconnect=connection + timedelta(hours=4),
                    kwh=12.0,
                    p_max_kw=6.0,
                )
            )
    return sessions


def _windows() -> list[dict]:
    """Dear 08:00–09:00 so a healthy schedule waits; the rest of the dwell is cheap."""
    days = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    return [
        {"name": "off_peak", "start": "00:00", "end": "08:00", "days": days, "price_per_kwh": 0.06},
        {"name": "on_peak", "start": "08:00", "end": "09:00", "days": days, "price_per_kwh": 0.40},
        {"name": "off_peak", "start": "09:00", "end": "16:00", "days": days, "price_per_kwh": 0.08},
        {"name": "mid_peak", "start": "16:00", "end": "24:00", "days": days, "price_per_kwh": 0.20},
    ]


def ignore_pilot(committed, epoch, active_index, delivered, sessions, cap_kw, dt, frac=0.2, extra_kwh=0.5) -> float:
    """One epoch: a fraction of cars charge extra kWh inside p_max and the site cap."""
    if not active_index or frac <= 0:
        return 0.0
    n_fault = max(1, int(round(frac * len(active_index))))
    extra_cost = 0.0
    for index in active_index[:n_fault]:
        cap_left = cap_kw - float(committed[:, epoch].sum())
        headroom = min(sessions[index].p_max_kw - float(committed[index, epoch]), cap_left)
        add_kw = min(headroom, extra_kwh / dt)
        if add_kw <= 1e-9:
            continue
        committed[index, epoch] += add_kw
        delivered[index] += add_kw * dt
        extra_cost += add_kw * dt * 0.2
    return extra_cost


_HEADLINE_M: float | None = None


def _headline_m() -> float:
    global _HEADLINE_M
    if _HEADLINE_M is None:
        import yaml

        from eci.config import REPO_ROOT

        raw = yaml.safe_load((REPO_ROOT / "configs" / "scheduler.yaml").read_text(encoding="utf-8"))
        _HEADLINE_M = float(raw["penalty_unmet_headline"])
    return _HEADLINE_M


def _loop(cap_kw: float) -> MPCLoop:
    return MPCLoop(
        slot_minutes=15,
        committed_k=1,
        penalty_unmet=_headline_m(),
        cap_kw=cap_kw,
        tariff_windows=_windows(),
        timezone_name="America/Los_Angeles",
        median_dwell=timedelta(hours=4),
    )


def _policy(state, slots, prices, cap, dt):
    return solve_epoch_fast(state, slots, prices, cap, dt, _headline_m())


def _bill(log) -> float:
    return log.schedule.costs(_headline_m())[0]


def run_ch01(sessions: list[ScenarioSession] | None = None) -> dict:
    sessions = sessions or synthetic_week()
    twin = _loop(40.0).run(sessions, _policy, "T3", "twin")
    faulty_sessions = drop_sessions(sessions, 0.10, seed=7)
    faulty = _loop(40.0).run(faulty_sessions, _policy, "T3", "CH-01")
    i2 = [item for item in check_invariants(faulty.schedule, faulty_sessions) if item.startswith("I2")]
    delta = (_bill(faulty) - _bill(twin)) / _bill(twin)
    passed = (not i2) and delta <= 0.02
    return {
        "case": "CH-01",
        "passed": passed,
        "cost_delta": delta,
        "i2_breaches": len(i2),
        "note": "drop 10% of sessions; cost must not rise by more than 2%; no phantom charge",
    }


def run_ch06(sessions: list[ScenarioSession] | None = None) -> dict:
    sessions = sessions or synthetic_week(n_days=1, per_day=8)
    alerts: list[str] = []
    cap = force_infeasible(40.0, 0.5)
    try:
        log = _loop(cap).run(sessions, _policy, "T3", "CH-06")
    except Exception as exc:  # noqa: BLE001 — the test is that the solver does not crash
        return {"case": "CH-06", "passed": False, "cost_delta": float("nan"), "i2_breaches": 0, "note": f"crash {exc}"}
    unmet = float(log.schedule.unmet_kwh.sum())
    if unmet > 1e-6:
        alerts.append(f"unmet_kwh={unmet:.4f} priced at M={_headline_m():.0f}")
    violations = check_invariants(log.schedule, sessions)
    passed = (not violations) and unmet > 1e-6 and bool(alerts)
    return {
        "case": "CH-06",
        "passed": passed,
        "cost_delta": unmet,
        "i2_breaches": sum(1 for item in violations if item.startswith("I2")),
        "note": "cap cut to 50%; soft unmet is the priced signal; " + "; ".join(alerts),
    }


def run_ch08(sessions: list[ScenarioSession] | None = None) -> dict:
    sessions = sessions or synthetic_week(n_days=1, per_day=8)
    twin = _loop(40.0).run(sessions, _policy, "T3", "twin")

    def hook(epoch, active_index, delivered, requirements, committed):
        if epoch != 0:
            return 0.0
        return ignore_pilot(committed, epoch, active_index, delivered, sessions, 40.0, 0.25)

    faulty = _loop(40.0).run(sessions, _policy, "T3", "CH-08", plant_hook=hook)
    recovery = _recovery_epochs(twin.epoch_costs, faulty.epoch_costs)
    violations = check_invariants(faulty.schedule, sessions)
    passed = recovery <= 2 and not violations
    return {
        "case": "CH-08",
        "passed": passed,
        "cost_delta": float(recovery),
        "i2_breaches": sum(1 for item in violations if item.startswith("I2")),
        "note": f"20% of cars take an extra 0.5 kWh on epoch 0; recovery epochs={recovery}",
    }


def _recovery_epochs(twin: list[float], faulty: list[float]) -> int:
    n = min(len(twin), len(faulty))
    for step in range(1, n):
        base = abs(twin[step])
        scale = base if base > 1e-6 else 1.0
        if abs(faulty[step] - twin[step]) / scale <= 0.01:
            return step
    return n


def run_chaos_suite() -> list[dict]:
    return [run_ch01(), run_ch06(), run_ch08()]
