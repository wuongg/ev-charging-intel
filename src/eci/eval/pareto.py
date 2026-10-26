"""ε-weighted peak vs energy cost. Scenario analysis on a 20-day window."""

from __future__ import annotations

from eci.eval.dispatch import _by_day
from eci.eval.scale import SCALE_NOTE, apply_util
from eci.eval.sweep import _load, _prepare, _sce, _score, headline_penalty
from eci.optimize.solve import solve_epoch

EPSILON_GRID = (0.0, 0.01, 0.1, 1.0, 10.0)
UTIL_GRID = (1.0, 4.0)
PARETO_SCENARIOS = tuple(range(10))
N_DAYS = 20
DLMP_EPSILON = 0.01


def take_days(sessions: list, n_days: int = N_DAYS) -> list:
    grouped = _by_day(sessions)
    keys = sorted(grouped)[:n_days]
    out = []
    for key in keys:
        out.extend(grouped[key])
    return out


def _peak_policy(penalty: float, peak_weight: float):
    def policy(state, slots, prices, cap, dt):
        return solve_epoch(state, slots, prices, cap, dt, penalty, peak_weight=peak_weight)

    return policy


def epsilon_sweep_cell(task: tuple[int, float, float]) -> dict:
    scenario_id, util, epsilon = task
    penalty = headline_penalty()
    fleet = take_days(_prepare(_load("bank_K50.parquet", scenario_id), util=1, impute=True), N_DAYS)
    fleet = apply_util(fleet, util) if util != 1.0 else fleet
    windows, demand = _sce()
    result = _score(fleet, _peak_policy(penalty, epsilon), "T2", "P3", 150.0, penalty, windows, demand)
    mean_shadow = 0.0
    n_bind = 0
    return {
        "scenario_id": scenario_id,
        "util": util,
        "epsilon": epsilon,
        "cost": result["total_cost"],
        "energy_cost": result["energy_cost"],
        "peak_kw": result["peak_kw"],
        "unmet_rate": result["unmet_rate"],
        "n_days": N_DAYS,
        "note": SCALE_NOTE + f" Window is the first {N_DAYS} local days of the scenario year.",
        "mean_cap_shadow": mean_shadow,
        "n_binding_shadow": n_bind,
    }


def dlmp_day(scenario_id: int = 0) -> list[dict]:
    """One local day, ε=0.01, cap-row duals of the committed slot."""
    from datetime import timedelta

    from eci.eval.dispatch import CLOCK
    from eci.optimize.mpc import MPCLoop

    penalty = headline_penalty()
    fleet = _prepare(_load("bank_K50.parquet", scenario_id), util=1, impute=True)
    grouped = _by_day(fleet)
    day = sorted(grouped)[len(grouped) // 2]
    sessions = grouped[day]
    windows, _demand = _sce()
    loop = MPCLoop(
        slot_minutes=15,
        committed_k=1,
        penalty_unmet=penalty,
        cap_kw=150.0,
        tariff_windows=windows,
        timezone_name=CLOCK,
        median_dwell=timedelta(hours=5.27),
    )
    log = loop.run(sessions, _peak_policy(penalty, DLMP_EPSILON), "T2", "P3")
    rows = []
    for entry in log.dual_log:
        duals = entry.get("cap_duals") or []
        if not duals:
            continue
        rows.append(
            {
                "epoch": entry["epoch"],
                "cap_dual": float(duals[0]),
                "peak_dual": float((entry.get("peak_duals") or [0.0])[0]),
                "scenario_id": scenario_id,
                "day": str(day),
                "epsilon": DLMP_EPSILON,
            }
        )
    return rows
