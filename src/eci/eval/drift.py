"""PSI on the input stream plus KS on residuals. Frozen bins from the reference year."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
from scipy.stats import ks_2samp

EPS = 1e-4
PSI_TRIGGER = 0.2
KS_P_TRIGGER = 0.05
N_BINS = 10


def frozen_quantile_edges(values: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Quantile edges computed once on the reference window. Never refit on test."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.array([-np.inf, np.inf], dtype=float)
    interior = np.quantile(finite, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    edges = np.concatenate(([-np.inf], np.unique(interior), [np.inf]))
    return edges.astype(float)


def _mass(values: np.ndarray, edges: np.ndarray, eps: float = EPS) -> np.ndarray:
    counts = np.histogram(np.asarray(values, dtype=float), bins=edges)[0].astype(float)
    total = counts.sum()
    if total <= 0:
        share = np.full(counts.shape, 1.0 / max(len(counts), 1))
    else:
        share = counts / total
    return np.maximum(share, eps)


def psi(expected: np.ndarray, actual: np.ndarray, edges: np.ndarray, eps: float = EPS) -> float:
    """Population Stability Index with an ε-floor so empty bins stay finite."""
    e = _mass(expected, edges, eps)
    a = _mass(actual, edges, eps)
    return float(np.sum((a - e) * np.log(a / e)))


def ks_residual_p(reference: np.ndarray, window: np.ndarray) -> float:
    """Two-sample KS on residuals. Empty windows are not a rejection."""
    ref = np.asarray(reference, dtype=float)
    win = np.asarray(window, dtype=float)
    ref = ref[np.isfinite(ref)]
    win = win[np.isfinite(win)]
    if ref.size < 8 or win.size < 8:
        return 1.0
    return float(ks_2samp(ref, win, alternative="two-sided", mode="auto").pvalue)


def psi_ks_detector(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    edges: np.ndarray,
    residual_ref: np.ndarray,
    residual_win: np.ndarray,
    psi_threshold: float = PSI_TRIGGER,
    ks_p_threshold: float = KS_P_TRIGGER,
    eps: float = EPS,
) -> dict:
    """One window. Trigger if PSI or KS fires. Bins stay the reference edges."""
    psi_value = psi(expected, actual, edges, eps=eps)
    ks_p = ks_residual_p(residual_ref, residual_win)
    triggered = bool(psi_value > psi_threshold or ks_p < ks_p_threshold)
    return {"psi": psi_value, "ks_p": ks_p, "triggered": triggered}


def daily_frame(hourly: pl.DataFrame, time_col: str = "ts_local_hour") -> pl.DataFrame:
    return (
        hourly.with_columns(pl.col(time_col).dt.date().alias("day"))
        .group_by("day")
        .agg(
            pl.col("kWh_delivered").sum().alias("daily_kwh"),
            pl.col("n_sessions").sum().alias("hourly_counts"),
        )
        .sort("day")
    )


def hour_of_week_residual(hourly: pl.DataFrame, time_col: str = "ts_local_hour") -> pl.DataFrame:
    """Residual versus the reference hour-of-week mean. The mean is fit later."""
    return hourly.with_columns(
        (pl.col(time_col).dt.weekday() * 24 + pl.col(time_col).dt.hour()).alias("how")
    )


def how_means(hourly: pl.DataFrame, time_col: str = "ts_local_hour") -> dict[int, float]:
    stamped = hour_of_week_residual(hourly, time_col)
    rows = stamped.group_by("how").agg(pl.col("kWh_delivered").mean().alias("mean")).to_dicts()
    return {int(row["how"]): float(row["mean"]) for row in rows}


def residuals(hourly: pl.DataFrame, means: dict[int, float], time_col: str = "ts_local_hour") -> np.ndarray:
    stamped = hour_of_week_residual(hourly, time_col)
    how = stamped["how"].to_numpy()
    y = stamped["kWh_delivered"].to_numpy()
    mu = np.array([means.get(int(key), 0.0) for key in how], dtype=float)
    return y - mu


def _in_range(hourly: pl.DataFrame, start: date, end: date, time_col: str = "ts_local_hour") -> pl.DataFrame:
    return hourly.filter((pl.col(time_col).dt.date() >= start) & (pl.col(time_col).dt.date() < end))


def month_windows(start: date, end: date) -> list[tuple[date, date]]:
    windows = []
    cursor = date(start.year, start.month, 1)
    while cursor < end:
        if cursor.month == 12:
            nxt = date(cursor.year + 1, 1, 1)
        else:
            nxt = date(cursor.year, cursor.month + 1, 1)
        windows.append((cursor, min(nxt, end)))
        cursor = nxt
    return windows


def is_true_shift_month(day: date, *, shift_start: date) -> bool:
    return day >= shift_start


def scan_windows(
    hourly: pl.DataFrame,
    *,
    reference_start: date,
    reference_end: date,
    scan_start: date,
    scan_end: date,
    shift_start: date,
    time_col: str = "ts_local_hour",
) -> list[dict]:
    """Monthly PSI/KS versus the frozen reference window."""
    ref = _in_range(hourly, reference_start, reference_end, time_col)
    daily_ref = daily_frame(ref, time_col)
    kwh_edges = frozen_quantile_edges(daily_ref["daily_kwh"].to_numpy())
    count_edges = frozen_quantile_edges(daily_ref["hourly_counts"].to_numpy())
    means = how_means(ref, time_col)
    res_ref = residuals(ref, means, time_col)
    rows = []
    for start, end in month_windows(scan_start, scan_end):
        window = _in_range(hourly, start, end, time_col)
        if window.height < 24 * 7:
            continue
        daily = daily_frame(window, time_col)
        kwh = psi_ks_detector(
            daily_ref["daily_kwh"].to_numpy(),
            daily["daily_kwh"].to_numpy(),
            edges=kwh_edges,
            residual_ref=res_ref,
            residual_win=residuals(window, means, time_col),
        )
        counts = psi_ks_detector(
            daily_ref["hourly_counts"].to_numpy(),
            daily["hourly_counts"].to_numpy(),
            edges=count_edges,
            residual_ref=res_ref,
            residual_win=residuals(window, means, time_col),
        )
        psi_value = max(float(kwh["psi"]), float(counts["psi"]))
        ks_p = min(float(kwh["ks_p"]), float(counts["ks_p"]))
        triggered = bool(kwh["triggered"] or counts["triggered"])
        rows.append(
            {
                "window": f"{start.isoformat()}/{end.isoformat()}",
                "psi": psi_value,
                "ks_p": ks_p,
                "triggered": triggered,
                "is_true_shift": is_true_shift_month(start, shift_start=shift_start),
                "psi_daily_kwh": float(kwh["psi"]),
                "psi_hourly_counts": float(counts["psi"]),
            }
        )
    return rows


def detection_delay_months(rows: list[dict], shift_start: date) -> float | None:
    """Months from the labeled shift until the first trigger on a true-shift window."""
    for row in rows:
        if not row["triggered"] or not row["is_true_shift"]:
            continue
        start = date.fromisoformat(str(row["window"]).split("/")[0])
        return (start.year - shift_start.year) * 12 + (start.month - shift_start.month)
    return None


def false_trigger_count(rows: list[dict]) -> int:
    return sum(1 for row in rows if row["triggered"] and not row["is_true_shift"])


def localize_hourly(hourly: pl.DataFrame, zone: str) -> pl.DataFrame:
    col = hourly["ts_local_hour"]
    tz = getattr(col.dtype, "time_zone", None)
    if tz:
        return hourly
    return hourly.with_columns(pl.col("ts_local_hour").dt.replace_time_zone(zone))


def acn_fallback_hourly(sessions_path, site: str = "caltech") -> pl.DataFrame:
    from eci.data.features import build_hourly_demand

    sessions = pl.read_parquet(sessions_path)
    hourly = build_hourly_demand(sessions, site)
    return localize_hourly(hourly, "America/Los_Angeles")


def _daily_from_hourly(hourly: pl.DataFrame, time_col: str = "ts_local_hour") -> pl.DataFrame:
    return (
        hourly.with_columns(pl.col(time_col).dt.date().alias("day"))
        .group_by("day")
        .agg(pl.col("kWh_delivered").sum().alias("y"))
        .sort("day")
    )


def _lgbm_daily(train_y: np.ndarray, train_days, pred_days) -> np.ndarray:
    from datetime import timedelta as _td

    import lightgbm as lgb

    train_feat = []
    for index, day in enumerate(train_days):
        if index < 7:
            continue
        train_feat.append(
            [train_y[index - 1], train_y[index - 7], day.weekday(), int(day.weekday() >= 5), day.month]
        )
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=0.5,
        n_estimators=80,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=10,
        subsample=0.8,
        subsample_freq=1,
        random_state=7,
        verbosity=-1,
        n_jobs=1,
    )
    model.fit(np.asarray(train_feat, dtype=float), np.asarray(train_y[7:], dtype=float))
    history = {day: value for day, value in zip(train_days, train_y, strict=True)}
    last = float(train_y[-1])
    pred_feat = []
    for day in pred_days:
        lag1 = history.get(day - _td(days=1), last)
        lag7 = history.get(day - _td(days=7), lag1)
        pred_feat.append([lag1, lag7, day.weekday(), int(day.weekday() >= 5), day.month])
    return np.asarray(model.predict(np.asarray(pred_feat, dtype=float)), dtype=float)


def _naive_daily(train_y: np.ndarray, train_days, pred_days) -> np.ndarray:
    from datetime import timedelta as _td

    history = {day: value for day, value in zip(train_days, train_y, strict=True)}
    last = float(train_y[-1])
    return np.asarray(
        [
            history.get(day - _td(days=7), history.get(day - _td(days=1), last))
            for day in pred_days
        ],
        dtype=float,
    )


def run_retrain_demo(
    hourly: pl.DataFrame,
    *,
    train_end: date,
    score_end: date | None = None,
    source: str,
) -> dict:
    """Fit LightGBM on the reference years; refit after a 14-day pinball-ratio trigger."""
    from eci.config import REPO_ROOT, provenance
    from eci.forecast.metrics import pinball_series

    daily = _daily_from_hourly(hourly)
    days = daily["day"].to_list()
    y = daily["y"].to_numpy()
    score_end = score_end or days[-1]
    train_idx = [i for i, day in enumerate(days) if day <= train_end]
    score_idx = [i for i, day in enumerate(days) if train_end < day <= score_end]
    if len(train_idx) < 40 or len(score_idx) < 20:
        raise ValueError("retrain demo needs a longer daily series")
    train_days = [days[i] for i in train_idx]
    train_y = y[train_idx]
    score_days = [days[i] for i in score_idx]
    score_y = y[score_idx]
    pred = _lgbm_daily(train_y, train_days, score_days)
    base = _naive_daily(train_y, train_days, score_days)
    model_pb = pinball_series(score_y, pred, 0.5)
    base_pb = pinball_series(score_y, base, 0.5)
    trigger_at = None
    streak = 0
    for index in range(len(score_days)):
        lo = max(0, index - 13)
        m = float(np.nanmean(model_pb[lo : index + 1]))
        b = float(np.nanmean(base_pb[lo : index + 1]))
        r = m / b if b > 1e-9 else 1.0
        if r > 1.0:
            streak += 1
        else:
            streak = 0
        if streak >= 14 and trigger_at is None:
            trigger_at = index
    if trigger_at is None:
        before = float(np.nanmean(model_pb))
        after = before
        improvement = 0.0
        trigger_day = ""
    else:
        before = float(np.nanmean(model_pb[: trigger_at + 1]))
        cut = trigger_at + 1
        new_train_days = train_days + score_days[:cut]
        new_train_y = np.concatenate([train_y, score_y[:cut]])
        rest_days = score_days[cut:]
        rest_y = score_y[cut:]
        if len(rest_days) < 7:
            after = before
            improvement = 0.0
        else:
            refit = _lgbm_daily(new_train_y, new_train_days, rest_days)
            after = float(np.nanmean(pinball_series(rest_y, refit, 0.5)))
            improvement = (before - after) / before if before > 1e-9 else 0.0
        trigger_day = score_days[trigger_at].isoformat()
    return {
        "before_trigger_pinball": before,
        "after_refit_pinball": after,
        "improvement_pct": 100.0 * improvement,
        "trigger_day": trigger_day,
        "n_score_days": len(score_days),
        "source": source,
        "note": "pinball tau=0.5; trigger = model/baseline > 1 for 14 consecutive days",
        **provenance(REPO_ROOT / "configs" / "forecast.yaml"),
    }
