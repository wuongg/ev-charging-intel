"""Parse, clean, and run the audit gate. Exit 1 when the gate fails."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pyarrow.parquet as pq

from eci.config import REPO_ROOT, provenance
from eci.data.audit import audit_gate, compare_reference
from eci.data.clean import clean_sessions
from eci.data.dictionary import build_data_dictionary
from eci.data.ingest import load_session_mirror
from eci.data.parse import parse_sessions


def _mirror_path() -> Path:
    raw = REPO_ROOT / "data" / "raw"
    preferred = [
        raw / "acndata_sessions.json",
        raw / "acndata_sessions.json.gz",
    ]
    for path in preferred:
        if path.exists():
            return path
    gz = sorted(raw.glob("acn_*.json.gz"))
    if gz:
        return gz[-1]
    raise FileNotFoundError("no session mirror under data/raw; run make data first")


def _write_parquet(frame, dest: Path, prov: dict) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    table = frame.to_arrow()
    meta = dict(table.schema.metadata or {})
    meta[b"eci_provenance"] = json.dumps(prov).encode("utf-8")
    table = table.replace_schema_metadata(meta)
    pq.write_table(table, dest)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    site_cfg = REPO_ROOT / "configs" / "site_caltech.yaml"
    prov = provenance(site_cfg)
    from eci.config import load_config

    timezone_name = load_config(site_cfg).model_dump().get("timezone") or "America/Los_Angeles"
    mirror = _mirror_path()
    sessions, truncated = load_session_mirror(mirror)
    print(f"sessions_recovered={len(sessions)}")
    print(f"truncated={truncated}")
    parsed = parse_sessions(sessions, timezone_name=timezone_name)
    cleaned, funnel = clean_sessions(parsed, timezone_name=timezone_name)
    report_path = REPO_ROOT / "results" / "audit_report.html"
    report = audit_gate(cleaned, funnel, report_path=report_path, provenance_info=prov)
    parquet_path = REPO_ROOT / "data" / "interim" / "sessions.parquet"
    _write_parquet(cleaned, parquet_path, prov)
    dictionary_path = REPO_ROOT / "docs" / "DATA_DICTIONARY.md"
    build_data_dictionary(cleaned, dictionary_path, prov, report.stats)

    print(f"quality_score={report.quality_score:.6f}")
    print(f"clock_skew_count={report.clock_skew_count}")
    print(f"clock_skew_rate={report.clock_skew_rate:.4%}")
    print(f"arrival_peak_local_hour={report.peak_hour}")
    print(f"gate={'PASS' if report.passed else 'FAIL'}")
    stats = report.stats
    print(f"long_dwell_24h_count={stats['long_dwell_24h_count']}")
    print(f"long_dwell_24h_rate={stats['long_dwell_24h_rate']:.4%}")
    print(f"long_dwell_48h_flag_count={stats['long_dwell_flag_count']}")
    print(f"user_id_missing_rate={stats['user_id_missing_rate']:.4%}")
    print(
        "multi_input_rate="
        f"{stats['multi_input_rate']:.4%} "
        f"({stats['multi_input_count']}/{stats['multi_input_with_inputs']} sessions with inputs)"
    )
    print(f"kwh_zero_count={stats['kwh_zero_count']}")
    print(f"kwh_zero_rate={stats['kwh_zero_rate']:.4%}")
    print("funnel")
    for row in funnel.to_dicts():
        print(
            f"  {row['stage']} {row['rule']} {row['action']} "
            f"in={row['n_in']} excluded={row['n_excluded']} "
            f"flagged={row['n_flagged']} out={row['n_out']}"
        )
    if stats["n"] >= 1000:
        print("reference_comparison")
        for line in compare_reference(stats):
            print(f"  {line}")
    for key, value in prov.items():
        print(f"{key}={value}")
    print(f"wrote {report_path}")
    print(f"wrote {parquet_path}")
    print(f"wrote {dictionary_path}")
    if not report.passed:
        for reason in report.reasons:
            print(f"GATE {reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
