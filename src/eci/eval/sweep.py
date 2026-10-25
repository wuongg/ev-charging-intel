"""M grid and cap × utilization grid. Both use T2v2 and the SCE bill."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import polars as pl
import yaml

from eci.config import REPO_ROOT, load_config
from eci.eval.dispatch import declared_inputs, run_policy_year, sessions_from_rows
from eci.optimize.impute import DwellSampler, sampler_from_parquet
from eci.optimize.policies import charge_on_arrival, cheapest_slots_greedy, earliest_deadline_first
from eci.optimize.solve import solve_epoch_fast
from eci.simulate.congestion import queue_edf

M_GRID = (10.0, 25.0, 100.0, 1000.0)
CAP_GRID = (60.0, 80.0, 100.0, 120.0, 150.0)
UTIL_GRID = (1, 2, 4)
N_PLUGS = 54

_SAMPLER: DwellSampler | None = None
_DECLARED: dict | None = None
_WINDOWS: list[dict] | None = None
_DEMAND: float | None = None


def _fast(penalty: float):
    def policy(state, slots, prices, cap, dt):
        return solve_epoch_fast(state, slots, prices, cap, dt, penalty)

    return policy


def _sampler() -> DwellSampler:
    global _SAMPLER
    if _SAMPLER is None:
        _SAMPLER = sampler_from_parquet(REPO_ROOT / "data" / "interim" / "sessions.parquet")
    return _SAMPLER


def _declared() -> dict:
    global _DECLARED
    if _DECLARED is None:
        _DECLARED = declared_inputs(REPO_ROOT / "data" / "interim" / "sessions.parquet")
    return _DECLARED


def _sce() -> tuple[list[dict], float]:
    global _WINDOWS, _DEMAND
    if _WINDOWS is None or _DEMAND is None:
        sce = load_config(REPO_ROOT / "configs" / "tariff_sce_ev4.yaml").model_dump()
        _WINDOWS = sce["windows"]
        _DEMAND = float(sce["demand_charge_per_kw_month"])
    return _WINDOWS, _DEMAND


def headline_penalty() -> float:
    raw = yaml.safe_load((REPO_ROOT / "configs" / "scheduler.yaml").read_text(encoding="utf-8"))
    return float(raw["penalty_unmet_headline"])


def _load(bank_name: str, scenario_id: int) -> list:
    bank = pl.read_parquet(REPO_ROOT / "scenarios" / bank_name)
    rows = bank.filter(pl.col("scenario_id") == scenario_id).to_dicts()
    return sessions_from_rows(rows, 208.0, 6.7, _declared())


def _with_copies(sessions: list, factor: int) -> list:
    if factor <= 1:
        return list(sessions)
    copies = []
    for copy in range(factor):
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
    return copies


def _prepare(sessions: list, util: int, impute: bool) -> list:
    fleet = _with_copies(sessions, util)
    if util > 1:
        fleet = queue_edf(fleet, n_plugs=N_PLUGS)
    if impute:
        fleet = _sampler().apply(fleet)
    return fleet


def _policies(penalty: float, include_p4: bool):
    rows = (
        ("P0", charge_on_arrival, "T2"),
        ("P1", cheapest_slots_greedy, "T2"),
        ("P2b", earliest_deadline_first, "T2"),
        ("P3", _fast(penalty), "T2"),
    )
    if include_p4:
        rows = (*rows, ("P4", _fast(penalty), "T3"))
    return rows


def _tariffs() -> list[tuple[str, list[dict], float]]:
    sce_windows, sce_demand = _sce()
    evn = load_config(REPO_ROOT / "configs" / "tariff_evn_2026.yaml").model_dump()
    two_part = yaml.safe_load((REPO_ROOT / "configs" / "tariff_evn_twopart.yaml").read_text(encoding="utf-8"))
    return [
        ("sce_ev4", sce_windows, sce_demand),
        (
            "evn_2026",
            evn["windows"],
            float(two_part["capacity_charge_vnd_per_kw_month"]["kv22_to_110"]),
        ),
    ]


def _score(sessions, policy, tier, name, cap, penalty, windows=None, demand=None) -> dict:
    if windows is None or demand is None:
        windows, demand = _sce()
    return run_policy_year(
        sessions,
        policy,
        tier,
        name,
        cap_kw=cap,
        windows=windows,
        median_dwell=timedelta(hours=5.27),
        penalty=penalty,
        demand_rate=demand,
    )


def _deadline_diagnosis(scenario_id: int, fleet: list, solved: dict) -> dict:
    """How much of the requested kWh sits outside the drawn deadline."""
    shorter = 0
    longer = 0
    imputed: list[float] = []
    truth: list[float] = []
    requested = 0.0
    outside_draw = 0.0
    outside_stay = 0.0
    for session in fleet:
        drawn = (session.requested_departure - session.connection).total_seconds() / 3600.0
        stayed = (session.disconnect - session.connection).total_seconds() / 3600.0
        imputed.append(drawn)
        truth.append(stayed)
        requested += session.kwh
        outside_draw += max(0.0, session.kwh - drawn * session.p_max_kw)
        outside_stay += max(0.0, session.kwh - stayed * session.p_max_kw)
        shorter += drawn < stayed
        longer += drawn > stayed
    imputed.sort()
    truth.sort()
    mid = len(fleet) // 2
    return {
        "scenario_id": scenario_id,
        "n_sessions": len(fleet),
        "frac_deadline_shorter": shorter / len(fleet),
        "frac_deadline_longer": longer / len(fleet),
        "median_imputed_h": imputed[mid],
        "median_true_h": truth[mid],
        "undeliverable_frac_imputed_window": outside_draw / requested if requested else 0.0,
        "undeliverable_frac_true_window": outside_stay / requested if requested else 0.0,
        "unmet_p0": solved["P0"]["unmet_rate"],
        "unmet_p1": solved["P1"]["unmet_rate"],
        "unmet_p3": solved["P3"]["unmet_rate"],
        "unmet_p4": solved["P4"]["unmet_rate"],
    }


def m_sensitivity_scenario(scenario_id: int) -> dict:
    """Five policies at each M. The schedule does not depend on M on this tariff.

    SCE prices are all below 0.27 USD/kWh, under every M in the grid, and the
    fast filler ignores the penalty. One replay is rescored at each M.
    """
    base = _load("bank_K50.parquet", scenario_id)
    fleet = _prepare(base, util=1, impute=True)
    solved = {}
    for name, policy, tier in _policies(M_GRID[0], include_p4=True):
        solved[name] = _score(fleet, policy, tier, name, 150.0, M_GRID[0])
    scored: list[dict] = []
    for penalty in M_GRID:
        totals = {}
        pending: list[dict] = []
        for name, result in solved.items():
            penalty_cost = result["unmet_kwh"] * penalty
            total = result["energy_cost"] + result["demand_cost"] + penalty_cost
            totals[name] = total
            pending.append(
                {
                    "scenario_id": scenario_id,
                    "M": penalty,
                    "policy": name,
                    "total_cost": total,
                    "penalty_cost": penalty_cost,
                    "penalty_share": (penalty_cost / total) if total else 0.0,
                    "unmet_rate": result["unmet_rate"],
                    "reduction_vs_p0": 0.0,
                }
            )
        baseline = totals["P0"]
        for row in pending:
            if row["policy"] != "P0" and baseline:
                row["reduction_vs_p0"] = (baseline - row["total_cost"]) / baseline
        scored.extend(pending)
    return {"rows": scored, "diagnosis": _deadline_diagnosis(scenario_id, fleet, solved)}


def comparison_scenario(scenario_id: int) -> list[dict]:
    """P0–P4 on both tariffs at the headline penalty. The schedule ignores M on these prices."""
    fleet = _prepare(_load("bank_K50.parquet", scenario_id), util=1, impute=True)
    penalty = headline_penalty()
    rows: list[dict] = []
    for tariff, windows, demand in _tariffs():
        solved = {}
        for name, policy, tier in _policies(penalty, include_p4=True):
            solved[name] = _score(fleet, policy, tier, name, 150.0, penalty, windows, demand)
        baseline = solved["P0"]["total_cost"]
        for name, result in solved.items():
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "tariff": tariff,
                    "policy": name,
                    "M": penalty,
                    "total_cost": result["total_cost"],
                    "energy_cost": result["energy_cost"],
                    "penalty_cost": result["penalty_cost"],
                    "unmet_rate": result["unmet_rate"],
                    "penalty_share": (result["penalty_cost"] / result["total_cost"]) if result["total_cost"] else 0.0,
                    "reduction_vs_p0": ((baseline - result["total_cost"]) / baseline) if baseline else 0.0,
                }
            )
    return rows


def declared_gap_scenario(scenario_id: int, months: set[int] | None = None) -> dict:
    """P3 at T2 with the declared departure versus the T3 oracle. Energy is kWhDelivered."""
    fleet = _sampler().apply(_load("bank_declared_K50.parquet", scenario_id))
    if months is not None:
        from zoneinfo import ZoneInfo

        zone = ZoneInfo("America/Los_Angeles")
        fleet = [session for session in fleet if session.connection.astimezone(zone).month in months]
    penalty = headline_penalty()
    if not fleet:
        return {"scenario_id": scenario_id, "gap": float("nan"), "n_sessions": 0}
    p3 = _score(fleet, _fast(penalty), "T2", "P3", 150.0, penalty)
    p4 = _score(fleet, _fast(penalty), "T3", "P4", 150.0, penalty)
    gap = (p3["total_cost"] - p4["total_cost"]) / p4["total_cost"] if p4["total_cost"] else float("nan")
    energy_p3 = p3["energy_cost"] + p3["penalty_cost"]
    energy_p4 = p4["energy_cost"] + p4["penalty_cost"]
    energy_gap = (energy_p3 - energy_p4) / energy_p4 if energy_p4 else float("nan")
    return {
        "scenario_id": scenario_id,
        "n_sessions": len(fleet),
        "cost_p3": p3["total_cost"],
        "cost_p4": p4["total_cost"],
        "gap": gap,
        "gap_energy_penalty": energy_gap,
        "unmet_p3": p3["unmet_rate"],
        "unmet_p4": p4["unmet_rate"],
        "penalty_share_p3": p3["penalty_cost"] / p3["total_cost"] if p3["total_cost"] else 0.0,
        "penalty_share_p4": p4["penalty_cost"] / p4["total_cost"] if p4["total_cost"] else 0.0,
        "M": penalty,
    }


def cap_sweep_scenario(scenario_id: int) -> list[dict]:
    """P0, P1, P3 on the cap × utilization grid. P4 only at 60 kW."""
    base = _load("bank_K50.parquet", scenario_id)
    penalty = headline_penalty()
    rows: list[dict] = []
    for util in UTIL_GRID:
        fleet = _prepare(base, util=util, impute=True)
        for cap in CAP_GRID:
            costs = {}
            names = ("P0", "P1", "P3", "P4") if cap == 60.0 else ("P0", "P1", "P3")
            for name, policy, tier in _policies(penalty, include_p4=True):
                if name not in names:
                    continue
                result = _score(fleet, policy, tier, name, cap, penalty)
                costs[name] = result
            p1 = costs["P1"]["total_cost"]
            p3 = costs["P3"]["total_cost"]
            gap = ((p1 - p3) / p1) if p1 else 0.0
            for name, result in costs.items():
                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "cap": cap,
                        "util": util,
                        "policy": name,
                        "cost": result["total_cost"],
                        "unmet_pct": result["unmet_rate"] * 100.0,
                        "plug_binding_fraction": result["plug_binding_fraction"],
                        "p3_vs_p1_gap": gap,
                        "penalty_share": (
                            result["penalty_cost"] / result["total_cost"] if result["total_cost"] else 0.0
                        ),
                    }
                )
    return rows
