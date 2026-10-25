"""Seasonal naive forecast. Point value is the same hour last week.

Quantiles are the trailing 28-day residual histogram added to the point
forecast. This is intentionally coarse; day 3 only needs a number.
"""

from __future__ import annotations

import statistics

import polars as pl


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def seasonal_naive(
    series: pl.DataFrame,
    m: int = 168,
    value_col: str = "kWh_delivered",
    time_col: str = "ts_local_hour",
) -> pl.DataFrame:
    """Predict y[t] with y[t-m]. Quantiles from residuals over the last 28 days.

    A series shorter than one week falls back to a 24-hour lag so the mini
    pipeline still emits a number.
    """
    values = [None if item is None else float(item) for item in series[value_col].to_list()]
    n = len(values)
    lag = m if n > m else 24
    point: list[float | None] = [None] * n
    for index in range(n):
        source = index - lag
        if source >= 0 and values[source] is not None:
            point[index] = values[source]

    window = 28 * 24
    residuals: list[float] = []
    for index in range(max(0, n - window), n):
        if point[index] is not None and values[index] is not None:
            residuals.append(values[index] - point[index])
    if residuals:
        q10 = _quantile(residuals, 0.10)
        q50 = statistics.median(residuals)
        q90 = _quantile(residuals, 0.90)
    else:
        q10 = q50 = q90 = 0.0

    def _shift(offset: float) -> list[float | None]:
        return [None if item is None else item + offset for item in point]

    return pl.DataFrame(
        {
            time_col: series[time_col],
            "yhat": point,
            "q10": _shift(q10),
            "q50": _shift(q50),
            "q90": _shift(q90),
        }
    )
