"""Audit gate. Q is the product of (1 - exclusion rate) over hard exclusions.

Clock-skew and the morning arrival peak are the pass/fail checks. A mirror that
peaks at 15:00–18:00 was parsed as UTC and fails the gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from eci.constants import (
    ARRIVAL_PEAK_HOURS,
    CLOCK_SKEW_RATE_MAX,
    CLOCK_SKEW_RATE_MIN,
    F_CLOCK_SKEW,
    F_LONG_DWELL,
    LONG_DWELL_REPORT_HOURS,
    REFERENCE_RELATIVE_TOLERANCE,
)

# Comparison targets from the day-1 brief for the truncated Caltech mirror.
# They are not universal population parameters.
REFERENCE_MIRROR = {
    "sessions": 13699,
    "clock_skew_count": 716,
    "clock_skew_rate": 716 / 13699,
    "long_dwell_24h_count": 143,
    "long_dwell_24h_rate": 143 / 13699,
    "user_id_missing_rate": 0.916,
    "multi_input_rate": 0.301,
}


@dataclass
class AuditReport:
    quality_score: float
    passed: bool
    reasons: list[str] = field(default_factory=list)
    clock_skew_count: int = 0
    clock_skew_rate: float = 0.0
    peak_hour: int | None = None
    stats: dict = field(default_factory=dict)


def quality_score(funnel: pl.DataFrame) -> float:
    """Π(1 - n_excluded/n_in) over EXCLUDE stages. Flags do not reduce Q."""
    score = 1.0
    exclusions = funnel.filter(pl.col("action") == "EXCLUDE")
    for n_in, n_excluded in zip(
        exclusions["n_in"].to_list(),
        exclusions["n_excluded"].to_list(),
        strict=True,
    ):
        if n_in <= 0:
            continue
        score *= 1.0 - (n_excluded / n_in)
    return score


def measure_stats(df: pl.DataFrame) -> dict:
    n = df.height
    if n == 0:
        return {
            "n": 0,
            "clock_skew_count": 0,
            "clock_skew_rate": 0.0,
            "long_dwell_flag_count": 0,
            "long_dwell_flag_rate": 0.0,
            "long_dwell_24h_count": 0,
            "long_dwell_24h_rate": 0.0,
            "user_id_missing_count": 0,
            "user_id_missing_rate": 0.0,
            "multi_input_count": 0,
            "multi_input_with_inputs": 0,
            "multi_input_rate": 0.0,
            "kwh_zero_count": 0,
            "kwh_zero_rate": 0.0,
            "peak_hour": None,
            "hour_counts": {},
        }
    dwell_hours = (
        (pl.col("disconnectTime_utc") - pl.col("connectionTime_utc")).dt.total_seconds() / 3600.0
    )
    long_24 = df.filter(dwell_hours > LONG_DWELL_REPORT_HOURS).height
    clock = int(df.filter(pl.col(F_CLOCK_SKEW)).height) if F_CLOCK_SKEW in df.columns else 0
    long_flag = int(df.filter(pl.col(F_LONG_DWELL)).height) if F_LONG_DWELL in df.columns else 0
    user_missing = df.filter(
        pl.col("userID").is_null() | (pl.col("userID").cast(pl.Utf8).str.strip_chars() == "")
    ).height
    with_inputs = df.filter(pl.col("n_user_inputs") > 0).height if "n_user_inputs" in df.columns else 0
    multi = df.filter(pl.col("n_user_inputs") > 1).height if "n_user_inputs" in df.columns else 0
    # 30.1% in the day-1 brief is among sessions that submitted userInputs, not all sessions.
    multi_rate = (multi / with_inputs) if with_inputs else 0.0
    kwh_zero = df.filter(pl.col("kWhDelivered") == 0).height
    hour_frame = (
        df.group_by(pl.col("connectionTime_local").dt.hour().alias("hour"))
        .len()
        .sort("len", descending=True)
    )
    hour_counts = {
        int(hour): int(count)
        for hour, count in zip(hour_frame["hour"].to_list(), hour_frame["len"].to_list(), strict=True)
        if hour is not None
    }
    peak = int(hour_frame["hour"][0]) if hour_frame.height else None
    return {
        "n": n,
        "clock_skew_count": clock,
        "clock_skew_rate": clock / n,
        "long_dwell_flag_count": long_flag,
        "long_dwell_flag_rate": long_flag / n,
        "long_dwell_24h_count": long_24,
        "long_dwell_24h_rate": long_24 / n,
        "user_id_missing_count": user_missing,
        "user_id_missing_rate": user_missing / n,
            "multi_input_count": multi,
            "multi_input_with_inputs": with_inputs,
            "multi_input_rate": multi_rate,
        "kwh_zero_count": kwh_zero,
        "kwh_zero_rate": kwh_zero / n,
        "peak_hour": peak,
        "hour_counts": hour_counts,
    }


def compare_reference(stats: dict) -> list[str]:
    """Print-ready lines. Relative error above 20% is a warning, not a hard fail."""
    checks = [
        ("sessions", stats["n"], REFERENCE_MIRROR["sessions"]),
        ("clock_skew_count", stats["clock_skew_count"], REFERENCE_MIRROR["clock_skew_count"]),
        (
            "long_dwell_24h_count",
            stats["long_dwell_24h_count"],
            REFERENCE_MIRROR["long_dwell_24h_count"],
        ),
        (
            "user_id_missing_rate",
            stats["user_id_missing_rate"],
            REFERENCE_MIRROR["user_id_missing_rate"],
        ),
        ("multi_input_rate", stats["multi_input_rate"], REFERENCE_MIRROR["multi_input_rate"]),
    ]
    lines: list[str] = []
    for name, actual, expected in checks:
        deviation = abs(actual - expected) / abs(expected) if expected else 0.0
        label = "WARNING" if deviation > REFERENCE_RELATIVE_TOLERANCE else "OK"
        lines.append(
            f"{label} {name}: actual={actual} reference={expected} relative_dev={deviation:.1%}"
        )
    return lines


def audit_gate(
    df: pl.DataFrame,
    funnel: pl.DataFrame,
    *,
    report_path: Path | None = None,
    provenance_info: dict | None = None,
) -> AuditReport:
    stats = measure_stats(df)
    score = quality_score(funnel)
    reasons: list[str] = []
    rate = stats["clock_skew_rate"]
    peak = stats["peak_hour"]
    if stats["n"] == 0:
        reasons.append("no sessions after cleaning")
    if not (CLOCK_SKEW_RATE_MIN < rate < CLOCK_SKEW_RATE_MAX):
        reasons.append(
            f"f_clock_skew rate {rate:.2%} is outside (2%, 10%) "
            f"(count={stats['clock_skew_count']})"
        )
    if peak not in ARRIVAL_PEAK_HOURS:
        hint = ""
        if peak in {15, 16, 17, 18}:
            hint = " UTC was probably stored as local time."
        reasons.append(f"arrival local-hour peak is {peak}, expected 7–10.{hint}")
    report = AuditReport(
        quality_score=score,
        passed=not reasons,
        reasons=reasons,
        clock_skew_count=stats["clock_skew_count"],
        clock_skew_rate=rate,
        peak_hour=peak,
        stats=stats,
    )
    if report_path is not None:
        write_audit_html(report_path, report, funnel, provenance_info or {})
    return report


def write_audit_html(
    path: Path,
    report: AuditReport,
    funnel: pl.DataFrame,
    provenance_info: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = report.stats
    hours = stats.get("hour_counts") or {}
    hour_rows = "".join(
        f"<tr><td>{hour:02d}</td><td>{hours.get(hour, 0)}</td></tr>" for hour in range(24)
    )
    provenance_rows = "".join(
        f"<li><code>{key}</code>: <code>{value}</code></li>" for key, value in provenance_info.items()
    )
    reasons = "".join(f"<li>{reason}</li>" for reason in report.reasons) or "<li>none</li>"
    status = "PASS" if report.passed else "FAIL"
    funnel_html = funnel.to_pandas().to_html(index=False)
    multi_count = stats.get("multi_input_count", 0)
    multi_base = stats.get("multi_input_with_inputs", 0)
    multi_rate = stats.get("multi_input_rate", 0)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>ECI audit gate — {status}</title>
<style>
body {{ font-family: Georgia, serif; margin: 2rem; color: #1c1917; }}
table {{ border-collapse: collapse; margin: 1rem 0; }}
td, th {{ border: 1px solid #d6d3d1; padding: 0.3rem 0.6rem; }}
.fail {{ color: #9f1239; }}
.pass {{ color: #166534; }}
</style>
</head>
<body>
<h1>Audit gate <span class="{status.lower()}">{status}</span></h1>
<p>Quality score Q = Π(1 − exclusion rate) over R1, R3, and V3. Flags stay in the data and do not reduce Q.</p>
<p><strong>Q = {report.quality_score:.6f}</strong></p>
<ul>
<li>f_clock_skew: {stats.get("clock_skew_count", 0)} ({stats.get("clock_skew_rate", 0):.2%})</li>
<li>f_long_dwell &gt; 48h: {stats.get("long_dwell_flag_count", 0)} ({stats.get("long_dwell_flag_rate", 0):.2%})</li>
<li>dwell &gt; 24h: {stats.get("long_dwell_24h_count", 0)} ({stats.get("long_dwell_24h_rate", 0):.2%})</li>
<li>userID missing: {stats.get("user_id_missing_count", 0)} ({stats.get("user_id_missing_rate", 0):.2%})</li>
<li>multi-input among sessions with userInputs: {multi_count} / {multi_base} ({multi_rate:.2%})</li>
<li>kWh == 0: {stats.get("kwh_zero_count", 0)} ({stats.get("kwh_zero_rate", 0):.2%})</li>
<li>arrival peak local hour: {report.peak_hour}</li>
</ul>
<h2>Gate</h2>
<ul>{reasons}</ul>
<h2>Provenance</h2>
<ul>{provenance_rows}</ul>
<h2>Funnel</h2>
{funnel_html}
<h2>Arrival hour (local)</h2>
<table><thead><tr><th>hour</th><th>sessions</th></tr></thead><tbody>
{hour_rows}
</tbody></table>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
