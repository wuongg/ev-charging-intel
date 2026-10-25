"""Expanding-window rolling-origin evaluation.

The embargo is 14 days and applies to the training prefix only. A fold's
``train_idx`` stops at ``origin − embargo``.

Feature availability is separate. Version 2 builds lags and rolling windows
from actuals strictly before the hour being forecast, including hours inside
the embargo and earlier hours of the horizon. Version 1 nulled every actual at
or after the cutoff, so every lag on the horizon was missing while seasonal
naive could still see a pre-cutoff actual. That was an information-asymmetric
comparison. The embargo length itself is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

import numpy as np
import polars as pl

from eci.forecast.naive import _quantile


@dataclass(frozen=True)
class Fold:
    """One origin. ``train_idx`` is every row strictly before the cutoff."""

    origin_idx: int
    cutoff_idx: int
    origin: datetime
    cutoff: datetime
    train_idx: np.ndarray
    test_idx: np.ndarray


def iter_folds(
    times: list[datetime],
    *,
    embargo_hours: int,
    origin_step_hours: int,
    horizon_hours: int,
    min_train_hours: int,
    origin_start: datetime,
    origin_end: datetime,
) -> list[Fold]:
    """Origins every ``origin_step_hours`` inside ``[origin_start, origin_end)``.

    The first eligible origin is ``min_train_hours + embargo_hours`` rows after
    the series starts, and the horizon must land inside the series.
    """
    n = len(times)
    folds: list[Fold] = []
    origin_idx = int(min_train_hours) + int(embargo_hours)
    step = int(origin_step_hours)
    horizon = int(horizon_hours)
    embargo = int(embargo_hours)
    while origin_idx + horizon <= n:
        origin = times[origin_idx]
        if origin_start <= origin < origin_end:
            cutoff_idx = origin_idx - embargo
            train_idx = np.arange(0, cutoff_idx, dtype=int)
            test_idx = np.arange(origin_idx, origin_idx + horizon, dtype=int)
            if train_idx.size and int(train_idx.max()) >= cutoff_idx:
                raise AssertionError("train index crossed the embargo")
            if test_idx.size and int(test_idx.min()) < origin_idx:
                raise AssertionError("test index starts before the origin")
            folds.append(
                Fold(
                    origin_idx=origin_idx,
                    cutoff_idx=cutoff_idx,
                    origin=origin,
                    cutoff=times[cutoff_idx],
                    train_idx=train_idx,
                    test_idx=test_idx,
                )
            )
        origin_idx += step
    return folds


def seasonal_naive_censored(
    y: np.ndarray,
    test_idx: np.ndarray,
    cutoff_idx: int,
    *,
    m: int = 168,
    residual_window_hours: int = 28 * 24,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> dict[float, np.ndarray]:
    """Seasonal naive that refuses any source hour at or after the cutoff."""
    values = np.asarray(y, dtype=float)
    point = np.empty(len(test_idx), dtype=float)
    for index, t in enumerate(test_idx):
        source = int(t) - m
        while source >= cutoff_idx:
            source -= m
        point[index] = values[source] if source >= 0 else np.nan
    residuals: list[float] = []
    start = max(0, cutoff_idx - residual_window_hours)
    for t in range(start, cutoff_idx):
        source = t - m
        while source >= cutoff_idx:
            source -= m
        if source >= 0 and np.isfinite(values[t]) and np.isfinite(values[source]):
            residuals.append(float(values[t] - values[source]))
    offsets = {float(tau): (_quantile(residuals, float(tau)) if residuals else 0.0) for tau in quantiles}
    return {tau: point + offset for tau, offset in offsets.items()}


def seasonal_naive_causal(
    y: np.ndarray,
    test_idx: np.ndarray,
    cutoff_idx: int,
    *,
    m: int = 168,
    residual_window_hours: int = 28 * 24,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> dict[float, np.ndarray]:
    """y[t − m] for every test hour. The source only has to be before t.

    Residual quantiles still come from the training prefix, before the cutoff.
    The embargo does not hide a lag that has already been observed.
    """
    values = np.asarray(y, dtype=float)
    point = np.empty(len(test_idx), dtype=float)
    for index, t in enumerate(test_idx):
        source = int(t) - m
        point[index] = values[source] if source >= 0 else np.nan
    residuals: list[float] = []
    start = max(0, cutoff_idx - residual_window_hours)
    for t in range(start, cutoff_idx):
        source = t - m
        if source >= 0 and np.isfinite(values[t]) and np.isfinite(values[source]):
            residuals.append(float(values[t] - values[source]))
    offsets = {float(tau): (_quantile(residuals, float(tau)) if residuals else 0.0) for tau in quantiles}
    return {tau: point + offset for tau, offset in offsets.items()}


def rolling_origin_cv(
    series: pl.DataFrame,
    model_fn: Callable[[pl.DataFrame, Fold], dict[float, np.ndarray]],
    *,
    embargo_hours: int,
    origin_step_hours: int,
    horizon_hours: int,
    min_train_hours: int,
    origin_start: datetime,
    origin_end: datetime,
    time_col: str = "ts_local_hour",
    value_col: str = "kWh_delivered",
) -> tuple[list[Fold], pl.DataFrame]:
    """Expanding window. ``model_fn`` receives the fold and must fit on ``train_idx`` only."""
    times = list(series[time_col].to_list())
    folds = iter_folds(
        times,
        embargo_hours=embargo_hours,
        origin_step_hours=origin_step_hours,
        horizon_hours=horizon_hours,
        min_train_hours=min_train_hours,
        origin_start=origin_start,
        origin_end=origin_end,
    )
    frames: list[pl.DataFrame] = []
    for fold in folds:
        if fold.train_idx.size and int(fold.train_idx.max()) >= fold.cutoff_idx:
            raise AssertionError("train index crossed the embargo")
        predicted = model_fn(series, fold)
        test = series.gather(fold.test_idx.tolist())
        columns: dict[str, object] = {
            time_col: test[time_col],
            "y": test[value_col],
            "origin": [fold.origin] * int(fold.test_idx.size),
        }
        for level, values in predicted.items():
            columns[f"q{int(round(float(level) * 100)):02d}"] = np.asarray(values, dtype=float)
        frames.append(pl.DataFrame(columns))
    if not frames:
        return folds, pl.DataFrame(
            schema={
                time_col: series.schema[time_col],
                "y": pl.Float64,
                "origin": series.schema[time_col],
            }
        )
    return folds, pl.concat(frames, how="vertical")


def localize_origin(value: datetime | str, sample: datetime) -> datetime:
    """Attach the series timezone to a config date that was stored without one."""
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if value.tzinfo is None:
        return value.replace(tzinfo=sample.tzinfo)
    return value.astimezone(sample.tzinfo)


def series_end_exclusive(times: list[datetime]) -> datetime:
    """One hour past the last stamp, so an origin on the last stamp is eligible."""
    return times[-1] + timedelta(hours=1)
