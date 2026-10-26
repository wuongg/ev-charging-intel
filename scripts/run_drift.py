"""Drift scan on Twente 2020–2024, or the labeled ACN 2018-vs-2020 proxy."""

from __future__ import annotations

import csv
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl

from eci.config import REPO_ROOT, provenance
from eci.data.download import download_twente
from eci.data.twente import TWENTE_NOTE, find_twente_file, hourly_demand, parse_twente_path
from eci.eval.drift import acn_fallback_hourly, detection_delay_months, false_trigger_count, scan_windows
from train_forecasts import retrain_trigger_demo

LOGGER = logging.getLogger(__name__)


def _load_hourly() -> tuple[pl.DataFrame, dict]:
    raw_dir = REPO_ROOT / "data" / "raw"
    path = find_twente_file(raw_dir)
    if path is None:
        download_twente(raw_dir)
        path = find_twente_file(raw_dir)
    if path is not None:
        sessions = parse_twente_path(path)
        hourly = hourly_demand(sessions)
        meta = {
            "source": "twente_asr",
            "license": "CC BY-NC-SA 4.0",
            "redistribute": False,
            "note": TWENTE_NOTE,
            "reference_start": date(2020, 6, 1),
            "reference_end": date(2021, 6, 1),
            "scan_start": date(2021, 6, 1),
            "shift_start": date(2021, 7, 1),
            "train_end": date(2021, 12, 31),
        }
        return hourly, meta
    LOGGER.warning("Twente download failed; using ACN 2018-vs-2020 proxy (labeled fallback)")
    hourly = acn_fallback_hourly(REPO_ROOT / "data" / "interim" / "sessions.parquet", "caltech")
    meta = {
        "source": "acn_2018_2020_proxy",
        "license": "ACN research dataset (existing mirror)",
        "redistribute": False,
        "note": (
            "FALLBACK. 4TU Twente/a.s.r. (CC BY-NC-SA) was unreachable. "
            "PSI/KS and retrain use ACN Caltech 2018 vs 2020 COVID as a labeled regime proxy."
        ),
        "reference_start": date(2018, 6, 1),
        "reference_end": date(2019, 6, 1),
        "scan_start": date(2019, 6, 1),
        "shift_start": date(2020, 3, 1),
        "train_end": date(2019, 12, 31),
    }
    return hourly, meta


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    hourly, meta = _load_hourly()
    last = hourly["ts_local_hour"].dt.date().max()
    rows = scan_windows(
        hourly,
        reference_start=meta["reference_start"],
        reference_end=meta["reference_end"],
        scan_start=meta["scan_start"],
        scan_end=last,
        shift_start=meta["shift_start"],
    )
    delay = detection_delay_months(rows, meta["shift_start"])
    false_n = false_trigger_count(rows)
    prov = provenance(REPO_ROOT / "configs" / "forecast.yaml")
    dest = REPO_ROOT / "results" / "drift_twente.csv"
    fieldnames = [
        "window",
        "psi",
        "ks_p",
        "triggered",
        "is_true_shift",
        "psi_daily_kwh",
        "psi_hourly_counts",
        "source",
        "license",
        "note",
        "detection_delay_months",
        "false_triggers",
        *prov.keys(),
    ]
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload.update(prov)
            payload["source"] = meta["source"]
            payload["license"] = meta["license"]
            payload["note"] = meta["note"]
            payload["detection_delay_months"] = delay
            payload["false_triggers"] = false_n
            writer.writerow(payload)
    print(f"wrote {dest} windows={len(rows)} delay_months={delay} false_triggers={false_n} source={meta['source']}")
    retrain_trigger_demo(hourly, train_end=meta["train_end"], source=meta["source"])
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
