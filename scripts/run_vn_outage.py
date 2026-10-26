"""P0/P3 × SCE headline M × S1 outage on/off on a frozen-bank slice."""

from __future__ import annotations

import csv
import sys
from datetime import timedelta
from zoneinfo import ZoneInfo

import polars as pl
import yaml

from eci.config import REPO_ROOT, load_config, provenance
from eci.eval.dispatch import sessions_from_rows
from eci.simulate.vn_scenarios import compare_outage, load_s1, sample_outages

LA = ZoneInfo("America/Los_Angeles")
N_DAYS = 45


def _sessions():
    bank = pl.read_parquet(REPO_ROOT / "scenarios" / "bank_K50.parquet")
    rows = bank.filter(pl.col("scenario_id") == 0).to_dicts()
    sessions = sessions_from_rows(rows, 208.0, 6.7)
    days = sorted({session.connection.astimezone(LA).date() for session in sessions})
    keep = set(days[:N_DAYS])
    return [session for session in sessions if session.connection.astimezone(LA).date() in keep]


def main() -> int:
    sessions = _sessions()
    params = load_s1()
    days = sorted({session.connection.astimezone(LA).date() for session in sessions})
    outages = sample_outages(days, params, seed=int(params.get("seed", 9)))
    sce = load_config(REPO_ROOT / "configs" / "tariff_sce_ev4.yaml").model_dump()
    penalty = float(
        yaml.safe_load((REPO_ROOT / "configs" / "scheduler.yaml").read_text(encoding="utf-8"))["penalty_unmet_headline"]
    )
    rows = compare_outage(
        sessions,
        cap_kw=150.0,
        windows=sce["windows"],
        median_dwell=timedelta(hours=5.2),
        penalty=penalty,
        demand_rate=float(sce["demand_charge_per_kw_month"]),
        outages=outages,
    )
    prov = provenance(REPO_ROOT / "configs" / "scenarios.yaml")
    dest = REPO_ROOT / "results" / "vn_outage_demo.csv"
    fieldnames = [
        "scenario",
        "outage_hours_served_missed",
        "cost",
        "unmet_in_outage",
        "recovery_cost",
        "unmet_kwh",
        "unmet_rate",
        "peak_kw",
        "i1_ok",
        "n_outages",
        "n_days",
        "n_sessions",
        "param_note",
        *prov.keys(),
    ]
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload.update(prov)
            payload["n_outages"] = len(outages)
            payload["n_days"] = len(days)
            payload["n_sessions"] = len(sessions)
            payload["param_note"] = params.get("note", "")
            writer.writerow(payload)
    print(f"wrote {dest} n_outages={len(outages)} days={len(days)} sessions={len(sessions)}")
    for row in rows:
        print(
            f"{row['scenario']} unmet={row['unmet_kwh']:.3f} cost={row['cost']:.2f} "
            f"i1={row['i1_ok']} missed_h={row['outage_hours_served_missed']:.2f}"
        )
    return 0 if all(row["i1_ok"] for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
