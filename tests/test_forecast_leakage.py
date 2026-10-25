"""Anti-leakage checks for the day-2 forecast stack."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

from eci.forecast.backtest import iter_folds, seasonal_naive_censored
from eci.forecast.lightgbm_q import (
    build_feature_frame,
    feature_column_names,
    predict_quantiles,
    train_lightgbm_quantile,
)
from eci.forecast.metrics import pinball_loss

LA = ZoneInfo("America/Los_Angeles")
LAGS = [1, 24, 48, 168, 336]
WINDOWS = [24, 168]
QUANTILES = [0.1, 0.5, 0.9]
PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "n_estimators": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 7,
    "n_jobs": 1,
}


def _hourly_frame(values: np.ndarray) -> pl.DataFrame:
    start = datetime(2018, 5, 7, tzinfo=LA)
    stamps = [start + timedelta(hours=index) for index in range(len(values))]
    return pl.DataFrame(
        {
            "ts_local_hour": stamps,
            "kWh_delivered": values.astype(float),
        }
    )


def test_rolling_feature_excludes_current_row() -> None:
    values = np.arange(1, 11, dtype=float)
    frame = build_feature_frame(_hourly_frame(values), lags=[1], windows=[2])
    # shift(1).rolling_mean(2) at index 4 uses y[2] and y[3], not y[4].
    assert frame["roll_mean_2"][4] == 3.5
    assert frame["lag_1"][4] == 4.0
    assert "kWh_delivered" not in feature_column_names([1], [2])


def test_no_fit_in_fold() -> None:
    start = datetime(2018, 1, 1, tzinfo=LA)
    times = [start + timedelta(hours=index) for index in range(800)]
    folds = iter_folds(
        times,
        embargo_hours=48,
        origin_step_hours=24,
        horizon_hours=24,
        min_train_hours=72,
        origin_start=times[0],
        origin_end=times[-1] + timedelta(hours=1),
    )
    assert len(folds) >= 1
    for fold in folds:
        assert int(fold.train_idx.max()) < fold.cutoff_idx
        assert fold.cutoff_idx == fold.origin_idx - 48
        assert fold.cutoff == times[fold.cutoff_idx]
        assert int(fold.test_idx.min()) >= fold.origin_idx
        assert set(fold.train_idx.tolist()).isdisjoint(fold.test_idx.tolist())
        assert int(fold.test_idx.min()) - int(fold.train_idx.max()) > 48


def test_shuffled_time_control() -> None:
    """A time-shuffled series must not beat seasonal-naive on the original series.

    If a feature contains the current actual, LightGBM predicts the shuffled
    test almost perfectly and pinball collapses. That is the leak.
    """
    rng = np.random.default_rng(7)
    n = 24 * 56
    hours = np.arange(n)
    weekend = ((hours // 24) % 7) >= 5
    original = 10 + 6 * np.sin(2 * np.pi * hours / 24) + 2 * weekend + rng.normal(0, 0.4, size=n)
    shuffled = original.copy()
    np.random.default_rng(11).shuffle(shuffled)

    test_from = n - 168
    names = feature_column_names(LAGS, WINDOWS)
    assert "kWh_delivered" not in names
    assert "_y_obs" not in names

    frame = build_feature_frame(_hourly_frame(shuffled), lags=LAGS, windows=WINDOWS)
    assert frame["lag_1"][test_from] == pytest.approx(float(shuffled[test_from - 1]))
    assert "kWh_delivered" not in names
    matrix = frame.select(names).to_numpy()
    target = shuffled
    train_idx = np.arange(336, test_from)
    models = train_lightgbm_quantile(
        matrix[train_idx],
        target[train_idx],
        names,
        QUANTILES,
        PARAMS,
    )
    predicted = predict_quantiles(models, matrix[test_from:])
    shuffled_pinball = float(
        np.mean([pinball_loss(target[test_from:], predicted[tau], tau) for tau in QUANTILES])
    )

    naive = seasonal_naive_censored(
        original,
        np.arange(test_from, n),
        test_from,
        m=168,
        quantiles=tuple(QUANTILES),
    )
    naive_pinball = float(np.mean([pinball_loss(original[test_from:], naive[tau], tau) for tau in QUANTILES]))
    assert naive_pinball > 0
    assert shuffled_pinball >= 0.8 * naive_pinball
