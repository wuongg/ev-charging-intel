"""Champion is the lowest mean pinball on the named evaluation window."""

from __future__ import annotations

import polars as pl


def pick_forecast(scores: pl.DataFrame, eval_window: str = "mirror_2018") -> dict:
    """Select on mean pinball. The paired test versus seasonal-naive is logged beside it.

    A model can be the champion and still fail to beat seasonal-naive. That
    loss is part of the result, not a reason to switch the rule.
    """
    window = scores.filter(pl.col("eval_window") == eval_window)
    pinball = window.filter(pl.col("metric") == "pinball")
    if pinball.height == 0:
        return {
            "eval_window": eval_window,
            "champion": None,
            "mean_pinball": None,
            "beats_naive": None,
            "p_value_vs_naive": None,
        }
    summary = (
        pinball.group_by("model")
        .agg(pl.col("value").mean().alias("mean_pinball"))
        .sort("mean_pinball")
    )
    champion = summary["model"][0]
    mean_pinball = float(summary["mean_pinball"][0])
    p_rows = window.filter((pl.col("model") == champion) & (pl.col("metric") == "pvalue_vs_naive"))
    p_value = None if p_rows.height == 0 or p_rows["value"][0] is None else float(p_rows["value"][0])
    naive = summary.filter(pl.col("model") == "seasonal_naive")
    naive_mean = None if naive.height == 0 else float(naive["mean_pinball"][0])
    beats = False
    if champion != "seasonal_naive" and naive_mean is not None and p_value is not None:
        beats = mean_pinball < naive_mean and p_value < 0.05
    return {
        "eval_window": eval_window,
        "champion": champion,
        "mean_pinball": mean_pinball,
        "naive_mean_pinball": naive_mean,
        "beats_naive": beats,
        "p_value_vs_naive": p_value,
    }
