"""Ugly end-to-end: P0 versus P3 on the first day of bank scenario 0."""

from __future__ import annotations

import csv
import sys
from datetime import timedelta
from zoneinfo import ZoneInfo

import polars as pl

from eci.config import REPO_ROOT, load_config, provenance
from eci.optimize.mpc import MPCLoop, ScenarioSession
from eci.optimize.policies import charge_on_arrival
from eci.optimize.solve import solve_epoch
from eci.simulate.congestion import plug_occupancy, queue_edf

LA = ZoneInfo("America/Los_Angeles")


def _policy_lp(penalty: float):
    def policy(state, slots, prices, cap, dt):
        return solve_epoch(state, slots, prices, cap, dt, penalty)

    return policy


def _sessions_for_day(bank: pl.DataFrame) -> tuple[list[ScenarioSession], str]:
    local = bank.filter(pl.col("scenario_id") == 0).with_columns(
        pl.col("connectionTime_utc").dt.convert_time_zone("America/Los_Angeles").alias("local_conn")
    )
    first = local.select(pl.col("local_conn").dt.date().min()).item()
    day = local.filter(pl.col("local_conn").dt.date() == first)
    sessions = []
    for row in day.iter_rows(named=True):
        connection = row["connectionTime_utc"]
        disconnect = row["disconnectTime_utc"]
        if disconnect <= connection or row["kWhDelivered"] <= 0:
            continue
        pilot = row.get("pilot_amp")
        p_max = 6.7
        if isinstance(pilot, (int, float)) and pilot > 1:
            p_max = float(pilot) * 208.0 / 1000.0
        sessions.append(
            ScenarioSession(
                session_id=str(row["sessionID"]),
                connection=connection,
                disconnect=disconnect,
                kwh=float(row["kWhDelivered"]),
                p_max_kw=p_max,
            )
        )
    return sessions, str(first)


def _median_dwell(sessions: pl.DataFrame) -> timedelta:
    dwell = (
        (pl.col("disconnectTime_utc") - pl.col("connectionTime_utc")).dt.total_seconds() / 3600.0
    )
    hours = sessions.select(dwell.alias("dwell_h")).filter(pl.col("dwell_h") > 0)["dwell_h"].median()
    return timedelta(hours=float(hours or 4.0))


def main() -> int:
    bank_path = REPO_ROOT / "scenarios" / "bank_K50.parquet"
    sessions_path = REPO_ROOT / "data" / "interim" / "sessions.parquet"
    if not bank_path.exists():
        print(f"missing {bank_path}", file=sys.stderr)
        return 1
    site = load_config(REPO_ROOT / "configs" / "site_caltech.yaml").model_dump()
    tariff = load_config(REPO_ROOT / "configs" / "tariff_sce_ev4.yaml").model_dump()
    scheduler = load_config(REPO_ROOT / "configs" / "scheduler.yaml").model_dump()
    penalty = float(scheduler["penalty_unmet"])
    cap = float(site["site_cap_kw"])
    bank = pl.read_parquet(bank_path)
    raw_day, day_label = _sessions_for_day(bank)
    queued = queue_edf(raw_day, n_plugs=int(site["evse_count"]))
    occupancy = plug_occupancy(queued)
    print(f"day={day_label} sessions={len(queued)} plug_binding={occupancy['binding_fraction']:.4f}")
    loop = MPCLoop(
        slot_minutes=int(scheduler["slot_minutes"]),
        committed_k=int(scheduler["committed_k"]),
        penalty_unmet=penalty,
        cap_kw=cap,
        tariff_windows=tariff["windows"],
        timezone_name=site["timezone"],
        median_dwell=_median_dwell(pl.read_parquet(sessions_path).filter(pl.col("siteID") == "0002")),
    )
    p0 = loop.run(queued, charge_on_arrival, "T3", "P0")
    p3 = loop.run(queued, _policy_lp(penalty), "T3", "P3")
    p0_cost = p0.schedule.costs(penalty)
    p3_cost = p3.schedule.costs(penalty)
    reduction = (p0_cost[0] - p3_cost[0]) / p0_cost[0] if p0_cost[0] else float("nan")
    print(f"cost_p0={p0_cost[0]:.6f} energy={p0_cost[1]:.6f} penalty={p0_cost[2]:.6f}")
    print(f"cost_p3={p3_cost[0]:.6f} energy={p3_cost[1]:.6f} penalty={p3_cost[2]:.6f}")
    print(f"reduction={reduction:.4%}")
    print(f"p3_running_peak_kw={p3.running_peak_kw:.3f}")
    if p3_cost[0] > p0_cost[0] + 1e-3:
        print("P3 cost exceeds P0; formulation needs a look before this result is quoted", file=sys.stderr)
        return 1
    prov = provenance(REPO_ROOT / "configs" / "scheduler.yaml")
    dest = REPO_ROOT / "results" / "e2e_day3.csv"
    note = (
        f"placeholder SCE prices; both policies use T3 actual departure and kWh; "
        f"day {day_label} of bank scenario 0; "
        f"plug binding {occupancy['binding_fraction']:.4f}; cap {cap} kW"
    )
    rows = []
    for name, log, costs in (("P0", p0, p0_cost), ("P3", p3, p3_cost)):
        rows.append(
            {
                "policy": name,
                "tier": log.tier,
                "cost": costs[0],
                "energy_cost": costs[1],
                "penalty_cost": costs[2],
                "reduction_vs_p0": 0.0 if name == "P0" else reduction,
                "day": day_label,
                "n_sessions": len(queued),
                "note": note,
                **prov,
            }
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
