"""Replace the T2v2 draw with one dwell quantile and rescore P3."""

from eci.eval.sweep import _load, _sampler, _score, headline_penalty
from eci.optimize.solve import solve_epoch_fast


def _fast(penalty: float):
    def policy(state, slots, prices, cap, dt):
        return solve_epoch_fast(state, slots, prices, cap, dt, penalty)

    return policy


def dial_scenario(task: tuple[int, float]) -> dict:
    scenario_id, tau = task
    penalty = headline_penalty()
    fleet = _sampler().apply_quantile(_load("bank_K50.parquet", scenario_id), tau)
    result = _score(fleet, _fast(penalty), "T2", "P3", 150.0, penalty)
    return {
        "scenario_id": scenario_id,
        "tau": tau,
        "M": penalty,
        "energy_cost": result["energy_cost"],
        "unmet_pct": result["unmet_rate"] * 100.0,
        "total_cost": result["total_cost"],
        "penalty_share": (result["penalty_cost"] / result["total_cost"]) if result["total_cost"] else 0.0,
    }
