"""LightGBM quantile regression for hourly site demand.

Rolling features are shifted by one hour before the window is applied, so the
current hour's actual never enters a lag or a rolling statistic. Holiday flags
are a fixed 2018–2021 federal list, not a calendar object that could see a
future holiday rule.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import polars as pl

# Observed US federal holidays, 2018–2021. Static on purpose.
US_FEDERAL_HOLIDAYS_2018_2021: frozenset[date] = frozenset(
    {
        date(2018, 1, 1),
        date(2018, 1, 15),
        date(2018, 2, 19),
        date(2018, 5, 28),
        date(2018, 7, 4),
        date(2018, 9, 3),
        date(2018, 10, 8),
        date(2018, 11, 12),
        date(2018, 11, 22),
        date(2018, 12, 25),
        date(2019, 1, 1),
        date(2019, 1, 21),
        date(2019, 2, 18),
        date(2019, 5, 27),
        date(2019, 7, 4),
        date(2019, 9, 2),
        date(2019, 10, 14),
        date(2019, 11, 11),
        date(2019, 11, 28),
        date(2019, 12, 25),
        date(2020, 1, 1),
        date(2020, 1, 20),
        date(2020, 2, 17),
        date(2020, 5, 25),
        date(2020, 7, 3),
        date(2020, 9, 7),
        date(2020, 10, 12),
        date(2020, 11, 11),
        date(2020, 11, 26),
        date(2020, 12, 25),
        date(2021, 1, 1),
        date(2021, 1, 18),
        date(2021, 2, 15),
        date(2021, 5, 31),
        date(2021, 6, 18),
        date(2021, 7, 5),
        date(2021, 9, 6),
        date(2021, 10, 11),
        date(2021, 11, 11),
        date(2021, 11, 25),
        date(2021, 12, 24),
    }
)

CALENDAR_FEATURES = ("hour", "dow", "is_weekend", "is_holiday")


def feature_column_names(lags: list[int], windows: list[int]) -> list[str]:
    """Column order passed to LightGBM. The raw target is not in this list."""
    names = [f"lag_{lag}" for lag in lags]
    for window in windows:
        names.append(f"roll_mean_{window}")
        names.append(f"roll_std_{window}")
    names.extend(CALENDAR_FEATURES)
    return names


def build_feature_frame(
    series: pl.DataFrame,
    *,
    lags: list[int],
    windows: list[int],
    value_col: str = "kWh_delivered",
    time_col: str = "ts_local_hour",
    cutoff: datetime | None = None,
    holidays: frozenset[date] = US_FEDERAL_HOLIDAYS_2018_2021,
) -> pl.DataFrame:
    """Feature rows aligned to ``series``.

    When ``cutoff`` is set, actuals at or after that timestamp become null
    before any lag or rolling window is computed. That is the information set
    at an origin: nothing from the embargo or the horizon is visible.
    """
    if cutoff is None:
        observed = pl.col(value_col)
    else:
        observed = pl.when(pl.col(time_col) < cutoff).then(pl.col(value_col)).otherwise(None)
    frame = series.with_columns(observed.alias("_y_obs"))
    exprs: list[pl.Expr] = [pl.col("_y_obs").shift(lag).alias(f"lag_{lag}") for lag in lags]
    for window in windows:
        shifted = pl.col("_y_obs").shift(1)
        exprs.append(shifted.rolling_mean(window_size=window, min_samples=window).alias(f"roll_mean_{window}"))
        exprs.append(shifted.rolling_std(window_size=window, min_samples=window).alias(f"roll_std_{window}"))
    exprs.extend(
        [
            pl.col(time_col).dt.hour().alias("hour"),
            pl.col(time_col).dt.weekday().alias("dow"),
            (pl.col(time_col).dt.weekday() >= 6).cast(pl.Int8).alias("is_weekend"),
            pl.col(time_col).dt.date().is_in(list(holidays)).cast(pl.Int8).alias("is_holiday"),
        ]
    )
    return frame.with_columns(exprs)


def train_lightgbm_quantile(
    features: np.ndarray,
    target: np.ndarray,
    feature_names: list[str],
    quantiles: list[float],
    params: dict,
) -> dict[float, object]:
    """Fit one quantile model per tau. ``params`` come from configs/forecast.yaml."""
    import lightgbm as lgb

    features = np.asarray(features, dtype=float)
    target = np.asarray(target, dtype=float)
    if features.ndim != 2:
        raise ValueError("features must be a 2d array")
    if features.shape[0] != target.shape[0]:
        raise ValueError("features and target must have the same number of rows")
    if features.shape[1] != len(feature_names):
        raise ValueError("feature_names length does not match the feature matrix")
    if "kWh_delivered" in feature_names or "_y_obs" in feature_names:
        raise ValueError("the raw target must not be a feature")
    finite = np.isfinite(target)
    features = features[finite]
    target = target[finite]
    if target.size == 0:
        raise ValueError("no finite training targets")

    models: dict[float, object] = {}
    for tau in quantiles:
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=float(tau),
            learning_rate=float(params["learning_rate"]),
            num_leaves=int(params["num_leaves"]),
            min_child_samples=int(params["min_child_samples"]),
            n_estimators=int(params["n_estimators"]),
            subsample=float(params["subsample"]),
            subsample_freq=1,
            colsample_bytree=float(params["colsample_bytree"]),
            random_state=int(params["seed"]),
            verbosity=-1,
            n_jobs=int(params.get("n_jobs", 1)),
        )
        model.fit(features, target, feature_name=list(feature_names))
        models[float(tau)] = model
    return models


def predict_quantiles(models: dict[float, object], features: np.ndarray) -> dict[float, np.ndarray]:
    """Predict every fitted quantile. Keys stay the training taus."""
    import pandas as pd

    matrix = np.asarray(features, dtype=float)
    predicted: dict[float, np.ndarray] = {}
    for tau, model in models.items():
        names = list(model.booster_.feature_name())
        frame = pd.DataFrame(matrix, columns=names)
        predicted[tau] = np.asarray(model.predict(frame), dtype=float)
    return predicted
