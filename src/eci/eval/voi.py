"""Value of telling the policy the truth, on a plant that already knows the plug."""

from __future__ import annotations

from eci.config import REPO_ROOT
from eci.eval.dispatch import sessions_from_rows
from eci.eval.sweep import _declared, _load, _policies, _prepare, _sampler, _score, _tariffs

# Commute assumption, labeled wherever the scaled figure is printed.
SESSIONS_PER_VEHICLE_YEAR = 200
FLEET_VEHICLES = 5000
M_GRID = (10.0, 25.0)


def _bill(result: dict, penalty: float) -> float:
    return result["energy_cost"] + result["demand_cost"] + result["unmet_kwh"] * penalty


def _declared_fleet(scenario_id: int, use_requested: bool) -> list:
    import polars as pl

    bank = pl.read_parquet(REPO_ROOT / "scenarios" / "bank_declared_K50.parquet")
    rows = bank.filter(pl.col("scenario_id") == scenario_id).to_dicts()
    sessions = sessions_from_rows(rows, 208.0, 6.7, _declared(), use_requested_kwh=use_requested)
    return _sampler().apply(sessions)


def _solve(fleet, tariff_windows, demand, penalty: float) -> dict[str, dict]:
    fast = _policies(penalty, include_p4=True)
    named = {name: (policy, tier) for name, policy, tier in fast}
    out = {}
    for key, tier_name in (("p0", "P0"), ("t2", "P3"), ("t3", "P4")):
        policy, tier = named[tier_name]
        # P4 in the table is the same fast filler at T3. Call it directly so a
        # requested-kWh fleet can be solved at T3 without a second policy object.
        if key == "t3":
            tier = "T3"
        out[key] = _score(fleet, policy, tier, tier_name, 150.0, penalty, tariff_windows, demand)
    return out


def _emit(scenario_id: int, bank: str, fleet, tariff: str, windows, demand, info: str) -> list[dict]:
    solved = _solve(fleet, windows, demand, M_GRID[0])
    rows = []
    for penalty in M_GRID:
        cost_p0 = _bill(solved["p0"], penalty)
        cost_t2 = _bill(solved["t2"], penalty)
        cost_t3 = _bill(solved["t3"], penalty)
        voi = cost_t2 - cost_t3
        rows.append(
            {
                "scenario_id": scenario_id,
                "bank": bank,
                "tariff": tariff,
                "M": penalty,
                "info": info,
                "n_sessions": len(fleet),
                "cost_p0": cost_p0,
                "cost_t2": cost_t2,
                "cost_t3": cost_t3,
                "voi": voi,
                "voi_fraction_of_p0": (voi / cost_p0) if cost_p0 else 0.0,
                "unmet_t2": solved["t2"]["unmet_rate"],
                "unmet_t3": solved["t3"]["unmet_rate"],
            }
        )
    return rows


def voi_scenario(scenario_id: int) -> list[dict]:
    """Imputed deadline versus the oracle, energy fixed at kWhDelivered. Frozen bank."""
    fleet = _prepare(_load("bank_K50.parquet", scenario_id), util=1, impute=True)
    rows: list[dict] = []
    for tariff, windows, demand in _tariffs():
        rows.extend(_emit(scenario_id, "bank_K50", fleet, tariff, windows, demand, "deadline_imputed"))
    return rows


def voi_ablation_scenario(scenario_id: int) -> list[dict]:
    """Declared bank, where both a departure and a kWhRequested exist."""
    oracle_energy = _declared_fleet(scenario_id, use_requested=False)
    declared_energy = _declared_fleet(scenario_id, use_requested=True)
    rows: list[dict] = []
    for tariff, windows, demand in _tariffs():
        belief = _solve(oracle_energy, windows, demand, M_GRID[0])
        requested = _solve(declared_energy, windows, demand, M_GRID[0])
        for penalty in M_GRID:
            deadline = _bill(belief["t2"], penalty) - _bill(belief["t3"], penalty)
            energy = _bill(requested["t3"], penalty) - _bill(belief["t3"], penalty)
            both = _bill(requested["t2"], penalty) - _bill(belief["t3"], penalty)
            base = _bill(belief["p0"], penalty)
            for component, value in (("deadline", deadline), ("energy", energy), ("both", both)):
                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "tariff": tariff,
                        "M": penalty,
                        "component": component,
                        "voi": value,
                        "voi_fraction_of_p0": (value / base) if base else 0.0,
                        "n_sessions": len(oracle_energy),
                    }
                )
    return rows
