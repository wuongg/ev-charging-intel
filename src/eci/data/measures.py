"""Hazard metrics measured on cleaned sessions.

ICED is a session-level proxy. This mirror has no per-minute power profile, so
the real ICED rate needs a later per-minute extract (path B1). Do not quote the
proxy as the minute-level ICED rate.
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import median

import polars as pl

from eci.config import REPO_ROOT, provenance
from eci.data.features import build_hourly_demand
from eci.data.parse import parse_rfc1123_utc

LOW_CONFIDENCE_N = 100
# kWh below this fraction of (dwell × station median kW) counts as the ICED proxy.
ICED_POWER_FRACTION = 0.25
ICED_MIN_DWELL_HOURS = 1.0
ICED_MIN_STATION_SESSIONS = 5


def _loads_snapshot(value: object) -> dict | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        import orjson

        loaded = orjson.loads(value)
        return loaded if isinstance(loaded, dict) else None
    return None


def early_unplug_compliance(df: pl.DataFrame) -> dict:
    """P(disconnect < requestedDeparture) on sessions with a decision-time snapshot.

    Benign: energy already meets kWhRequested (finished early, not an SLA miss).
    Violating: left before the requested departure with kWhDelivered < kWhRequested.
    Gap minutes are requestedDeparture minus disconnect (positive means early).
    """
    gaps: list[float] = []
    early_gaps: list[float] = []
    n_early = 0
    n_benign = 0
    n_violating = 0
    n_unclassified = 0
    for row in df.select("disconnectTime_utc", "kWhDelivered", "userInputs_last").iter_rows(named=True):
        snapshot = _loads_snapshot(row["userInputs_last"])
        disconnect = row["disconnectTime_utc"]
        if snapshot is None or disconnect is None or "requestedDeparture" not in snapshot:
            continue
        try:
            requested = parse_rfc1123_utc(str(snapshot["requestedDeparture"]))
        except (TypeError, ValueError):
            continue
        gap_min = (requested - disconnect).total_seconds() / 60.0
        gaps.append(gap_min)
        if disconnect < requested:
            n_early += 1
            early_gaps.append(gap_min)
            requested_kwh = snapshot.get("kWhRequested")
            delivered = row["kWhDelivered"]
            if requested_kwh is None or delivered is None:
                n_unclassified += 1
            elif delivered >= float(requested_kwh):
                n_benign += 1
            else:
                n_violating += 1
    n = len(gaps)
    return {
        "n": n,
        "p_early": (n_early / n) if n else None,
        "p_benign": (n_benign / n) if n else None,
        "p_violating": (n_violating / n) if n else None,
        "n_early": n_early,
        "n_benign": n_benign,
        "n_violating": n_violating,
        "n_unclassified": n_unclassified,
        "median_gap_min": median(gaps) if gaps else None,
        "median_gap_min_early": median(early_gaps) if early_gaps else None,
        "low_confidence": n < LOW_CONFIDENCE_N,
    }


def overstay_rate(df: pl.DataFrame) -> dict:
    """P(disconnect − doneCharging > 15 min) and the >1h tail.

    Rows flagged f_clock_skew are removed first. doneCharging after disconnect
    is the known clock defect and is not an overstay.
    """
    kept = df.filter(~pl.col("f_clock_skew"))
    gap_min = (
        (pl.col("disconnectTime_utc") - pl.col("doneChargingTime_utc")).dt.total_seconds() / 60.0
    )
    valid = kept.filter(pl.col("doneChargingTime_utc").is_not_null()).with_columns(gap_min.alias("gap_min"))
    n = valid.height
    if n == 0:
        return {"n": 0, "p_over_15min": None, "p_over_1h": None, "low_confidence": True}
    p15 = valid.filter(pl.col("gap_min") > 15).height / n
    p60 = valid.filter(pl.col("gap_min") > 60).height / n
    return {
        "n": n,
        "p_over_15min": p15,
        "p_over_1h": p60,
        "low_confidence": n < LOW_CONFIDENCE_N,
    }


def iced_proxy_session_level(df: pl.DataFrame) -> dict:
    """Session-level stand-in for ICED.

    PROXY, not the minute-level ICED rate. A session counts when its delivered
    energy is below 25% of dwell hours times that station's median kW, the dwell
    is at least 1 hour, and the station has at least 5 reference sessions.
    Reference power uses sessions with dwell in [0.25h, 24h] and positive kWh.
    The real rate needs per-minute profiles.
    """
    dwell_h = (pl.col("disconnectTime_utc") - pl.col("connectionTime_utc")).dt.total_seconds() / 3600.0
    framed = df.with_columns(dwell_h.alias("dwell_h")).filter(pl.col("dwell_h") > 0)
    reference = framed.filter(
        (pl.col("dwell_h") >= 0.25)
        & (pl.col("dwell_h") <= 24)
        & (pl.col("kWhDelivered") > 0)
    ).with_columns((pl.col("kWhDelivered") / pl.col("dwell_h")).alias("power_kw"))
    station = reference.group_by("stationID").agg(
        pl.col("power_kw").median().alias("median_kw"),
        pl.len().alias("n_ref"),
    )
    joined = framed.join(station, on="stationID", how="left")
    eligible = joined.filter(
        (pl.col("dwell_h") >= ICED_MIN_DWELL_HOURS) & (pl.col("n_ref") >= ICED_MIN_STATION_SESSIONS)
    )
    n = eligible.height
    if n == 0:
        return {"n": 0, "p_iced_proxy": None, "low_confidence": True}
    iced = eligible.filter(
        pl.col("kWhDelivered") < ICED_POWER_FRACTION * pl.col("dwell_h") * pl.col("median_kw")
    ).height
    return {
        "n": n,
        "p_iced_proxy": iced / n,
        "n_iced": iced,
        "low_confidence": n < LOW_CONFIDENCE_N,
    }


def overdispersion_test(hourly: pl.DataFrame) -> dict:
    """Variance-to-mean ratio of hourly session counts, by hour of day.

    A Poisson arrival process has ratio 1. A ratio well above 1 means the
    week-3 IPP should be overdispersed. Ratios use sample variance (ddof=1).
    """
    if hourly.height == 0:
        return {"n_hours": 0, "mean_ratio": None, "max_ratio": None, "low_confidence": True, "by_hour": {}}
    grouped = (
        hourly.with_columns(pl.col("ts_local_hour").dt.hour().alias("hour"))
        .group_by("hour")
        .agg(
            pl.col("n_sessions").mean().alias("mean_count"),
            pl.col("n_sessions").var().alias("var_count"),
            pl.len().alias("n_obs"),
        )
        .sort("hour")
    )
    by_hour: dict[int, float | None] = {}
    ratios: list[float] = []
    for row in grouped.iter_rows(named=True):
        mean_count = row["mean_count"]
        var_count = row["var_count"]
        ratio = None
        if mean_count and mean_count > 0 and var_count is not None:
            ratio = float(var_count) / float(mean_count)
            ratios.append(ratio)
        by_hour[int(row["hour"])] = ratio
    n_hours = hourly.height
    mean_ratio = sum(ratios) / len(ratios) if ratios else None
    max_ratio = max(ratios) if ratios else None
    max_hour = None
    if max_ratio is not None:
        max_hour = max(by_hour, key=lambda hour: by_hour[hour] or -1)
    return {
        "n_hours": n_hours,
        "n_clock_hours": len(ratios),
        "mean_ratio": mean_ratio,
        "max_ratio": max_ratio,
        "max_hour": max_hour,
        "by_hour": by_hour,
        "poisson_inadequate": bool(mean_ratio is not None and mean_ratio > 2),
        "low_confidence": n_hours < LOW_CONFIDENCE_N,
    }


def iced_true_rate(df: pl.DataFrame) -> dict:
    """Time-weighted share of plugged time with charging current below 1 A.

    Weights are session dwell hours. This is the minute-level rate from path B1.
    Sessions without ``iced_fraction`` are left out of the denominator.
    """
    if "iced_fraction" not in df.columns:
        return {"n": 0, "p_iced": None, "low_confidence": True}
    dwell_h = (pl.col("disconnectTime_utc") - pl.col("connectionTime_utc")).dt.total_seconds() / 3600.0
    valid = df.filter(pl.col("iced_fraction").is_not_null()).with_columns(dwell_h.alias("dwell_h"))
    valid = valid.filter(pl.col("dwell_h") > 0)
    n = valid.height
    if n == 0:
        return {"n": 0, "p_iced": None, "low_confidence": True}
    weight = float(valid["dwell_h"].sum())
    weighted = float((valid["iced_fraction"] * valid["dwell_h"]).sum())
    return {
        "n": n,
        "p_iced": (weighted / weight) if weight else None,
        "low_confidence": n < LOW_CONFIDENCE_N,
    }


def hazard_rows(sessions: pl.DataFrame, hourly: pl.DataFrame) -> list[dict]:
    """Flat metric rows for results/hazard_metrics.csv."""
    early = early_unplug_compliance(sessions)
    overstay = overstay_rate(sessions)
    iced = iced_proxy_session_level(sessions)
    true_iced = iced_true_rate(sessions)
    dispersion = overdispersion_test(hourly)
    iced_note = "minute-level, from B1"
    if true_iced["p_iced"] and iced["p_iced_proxy"] is not None:
        relative = abs(iced["p_iced_proxy"] - true_iced["p_iced"]) / true_iced["p_iced"]
        if relative > 0.5:
            iced_note += f"; proxy differs by {relative:.0%} relative"
    rows = [
        _row(
            "early_unplug_rate",
            early["p_early"],
            early["n"],
            early["low_confidence"],
            "P(disconnect < requestedDeparture) among snapshots with requestedDeparture",
        ),
        _row(
            "early_unplug_benign_rate",
            early["p_benign"],
            early["n"],
            early["low_confidence"],
            "Early and kWhDelivered >= kWhRequested, over the same denominator",
        ),
        _row(
            "early_unplug_violating_rate",
            early["p_violating"],
            early["n"],
            early["low_confidence"],
            "Early and kWhDelivered < kWhRequested, over the same denominator",
        ),
        _row(
            "early_unplug_median_gap_min",
            early["median_gap_min"],
            early["n"],
            early["low_confidence"],
            "Median of requestedDeparture minus disconnect, in minutes",
        ),
        _row(
            "overstay_rate_15min",
            overstay["p_over_15min"],
            overstay["n"],
            overstay["low_confidence"],
            "P(disconnect - doneCharging > 15 min) after dropping f_clock_skew",
        ),
        _row(
            "overstay_rate_1h",
            overstay["p_over_1h"],
            overstay["n"],
            overstay["low_confidence"],
            "P(disconnect - doneCharging > 1h) after dropping f_clock_skew",
        ),
        _row(
            "iced_proxy_rate",
            iced["p_iced_proxy"],
            iced["n"],
            iced["low_confidence"],
            "PROXY: kWh < 0.25 * dwell_h * station median kW. Not minute-level ICED.",
        ),
        _row(
            "iced_true_rate",
            true_iced["p_iced"],
            true_iced["n"],
            true_iced["low_confidence"],
            iced_note,
        ),
        _row(
            "overdispersion_mean_var_mean_ratio",
            dispersion["mean_ratio"],
            dispersion["n_hours"],
            dispersion["low_confidence"],
            "Mean across clock hours of var/mean of n_sessions. >>1 rejects Poisson.",
        ),
        _row(
            "overdispersion_max_var_mean_ratio",
            dispersion["max_ratio"],
            dispersion["n_hours"],
            dispersion["low_confidence"],
            f"Max hour-of-day var/mean, hour={dispersion['max_hour']}",
        ),
    ]
    return rows


def _row(metric: str, value: float | None, n: int, low_confidence: bool, detail: str) -> dict:
    return {
        "metric": metric,
        "value": value,
        "n": n,
        "low_confidence": bool(low_confidence),
        "detail": detail,
    }


def write_hazard_metrics(
    sessions: pl.DataFrame,
    hourly: pl.DataFrame,
    dest: Path | None = None,
    dictionary_path: Path | None = None,
    prov: dict | None = None,
) -> list[dict]:
    """Write the csv and refresh the Measured hazards section of the dictionary."""
    rows = hazard_rows(sessions, hourly)
    if prov is None:
        prov = provenance(REPO_ROOT / "configs" / "site_caltech.yaml")
    for row in rows:
        row.update(prov)
    path = dest or (REPO_ROOT / "results" / "hazard_metrics.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["metric", "value", "n", "low_confidence", "detail", *prov.keys()]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    upsert_hazard_section(dictionary_path or (REPO_ROOT / "docs" / "DATA_DICTIONARY.md"), rows)
    return rows


def hazard_section_body(rows: list[dict]) -> str:
    lines = [
        "Rates below are measured by `eci.data.measures` on the cleaned mirror.",
        "ICED is a session-level proxy; the minute-level rate needs per-minute profiles.",
        "",
        "| metric | value | n | low_confidence | detail |",
        "|---|---:|---:|---|---|",
    ]
    for row in rows:
        value = "NA" if row["value"] is None else f"{row['value']:.6g}"
        lines.append(
            f"| `{row['metric']}` | {value} | {row['n']} | {row['low_confidence']} | {row['detail']} |"
        )
    if rows:
        lines.extend(
            [
                "",
                f"- git_sha: `{rows[0].get('git_sha', '')}`",
                f"- config_hash: `{rows[0].get('config_hash', '')}`",
                f"- data_hash: `{rows[0].get('data_hash', '')}`",
                f"- utc_now: `{rows[0].get('utc_now', '')}`",
            ]
        )
    return "\n".join(lines)


def upsert_hazard_section(dictionary_path: Path, rows: list[dict]) -> None:
    """Insert or replace the Measured hazards section, leaving the rest of the file."""
    end = "## TODO verify before any cost or capacity claim"
    block = "## Measured hazards\n\n" + hazard_section_body(rows) + "\n\n"
    path = Path(dictionary_path)
    text = path.read_text(encoding="utf-8") if path.exists() else f"# Data dictionary\n\n{end}\n"
    start = "## Measured hazards"
    if start in text and end in text:
        pre, rest = text.split(start, 1)
        _, post = rest.split(end, 1)
        text = pre.rstrip() + "\n\n" + block + end + post
    elif end in text:
        text = text.replace(end, block + end, 1)
    else:
        text = text.rstrip() + "\n\n" + block
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def print_hazard_rows(rows: list[dict]) -> None:
    for row in rows:
        flag = " LOW_CONFIDENCE" if row["low_confidence"] or row["n"] < LOW_CONFIDENCE_N else ""
        value = "NA" if row["value"] is None else f"{row['value']:.6g}"
        print(f"{row['metric']}={value} n={row['n']}{flag}")


def main() -> int:
    sessions = pl.read_parquet(REPO_ROOT / "data" / "interim" / "sessions.parquet")
    hourly = build_hourly_demand(sessions, "caltech")
    rows = write_hazard_metrics(sessions, hourly)
    print_hazard_rows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
