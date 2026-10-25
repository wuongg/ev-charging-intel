"""Fit the day-2 forecast stack on the Caltech hourly series and write scores.

Chronos is not a member of this run. chronos-forecasting depends on torch, so
the member is dropped and the ensemble is seasonal-naive plus LightGBM.
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime

import numpy as np
import polars as pl

from eci.config import REPO_ROOT, load_config, provenance
from eci.data.features import build_hourly_demand
from eci.data.measures import print_hazard_rows, write_hazard_metrics
from eci.forecast.backtest import (
    iter_folds,
    localize_origin,
    rolling_origin_cv,
    seasonal_naive_censored,
    series_end_exclusive,
)
from eci.forecast.chronos import chronos_zero_shot
from eci.forecast.ensemble import equal_weight_quantiles, rearrange
from eci.forecast.lightgbm_q import (
    build_feature_frame,
    feature_column_names,
    predict_quantiles,
    train_lightgbm_quantile,
)
from eci.forecast.metrics import (
    empirical_coverage,
    mae,
    paired_moving_block_bootstrap,
    pinball_loss,
    pinball_series,
    quantile_crossing_rate,
)
from eci.forecast.select import pick_forecast

VALUE_COL = "kWh_delivered"
TIME_COL = "ts_local_hour"


def _params(cfg) -> dict:
    raw = cfg.lightgbm
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()
    return dict(raw)


def _naive_fn(cfg):
    quantiles = tuple(float(tau) for tau in cfg.quantiles)
    period = int(cfg.seasonal_period_hours)
    residual_window = int(cfg.residual_window_days) * 24

    def predict(series: pl.DataFrame, fold) -> dict[float, np.ndarray]:
        y = series[VALUE_COL].to_numpy()
        return seasonal_naive_censored(
            y,
            fold.test_idx,
            fold.cutoff_idx,
            m=period,
            residual_window_hours=residual_window,
            quantiles=quantiles,
        )

    return predict


def _lightgbm_fn(cfg, params: dict | None = None):
    quantiles = [float(tau) for tau in cfg.quantiles]
    lags = [int(lag) for lag in cfg.lags]
    windows = [int(window) for window in cfg.rolling_windows]
    names = feature_column_names(lags, windows)
    fitted = dict(params or _params(cfg))

    def predict(series: pl.DataFrame, fold) -> dict[float, np.ndarray]:
        frame = build_feature_frame(
            series,
            lags=lags,
            windows=windows,
            value_col=VALUE_COL,
            time_col=TIME_COL,
            cutoff=fold.cutoff,
        )
        matrix = frame.select(names).to_numpy()
        target = np.asarray(frame["_y_obs"].to_numpy(), dtype=float)
        train_target = target[fold.train_idx]
        train_matrix = matrix[fold.train_idx]
        models = train_lightgbm_quantile(train_matrix, train_target, names, quantiles, fitted)
        return predict_quantiles(models, matrix[fold.test_idx])

    return predict


def _ensemble_fn(cfg):
    naive = _naive_fn(cfg)
    lightgbm = _lightgbm_fn(cfg)

    def predict(series: pl.DataFrame, fold) -> dict[float, np.ndarray]:
        averaged = equal_weight_quantiles([naive(series, fold), lightgbm(series, fold)])
        return rearrange(averaged)

    return predict


def _qcol(tau: float) -> str:
    return f"q{int(round(tau * 100)):02d}"


def _mean_loss(frame: pl.DataFrame, quantiles: list[float]) -> np.ndarray:
    y = frame["y"].to_numpy()
    losses = [pinball_series(y, frame[_qcol(tau)].to_numpy(), tau) for tau in quantiles]
    return np.nanmean(np.vstack(losses), axis=0)


def _score_rows(
    frame: pl.DataFrame,
    model: str,
    window: str,
    quantiles: list[float],
    note: str,
) -> list[dict]:
    y = frame["y"].to_numpy()
    rows: list[dict] = []
    preds = {tau: frame[_qcol(tau)].to_numpy() for tau in quantiles}
    n = int(np.isfinite(y).sum())
    for tau in quantiles:
        rows.append(
            {
                "eval_window": window,
                "model": model,
                "quantile": f"{tau:.1f}",
                "metric": "pinball",
                "value": pinball_loss(y, preds[tau], tau),
                "n": n,
                "note": note,
            }
        )
        rows.append(
            {
                "eval_window": window,
                "model": model,
                "quantile": f"{tau:.1f}",
                "metric": "mae",
                "value": mae(y, preds[tau]),
                "n": n,
                "note": note,
            }
        )
    coverage = empirical_coverage(y, preds[0.1], preds[0.9])
    coverage_note = note
    if np.isfinite(coverage) and not (0.70 <= coverage <= 0.90):
        coverage_note = f"{note} OUTSIDE [0.70, 0.90]".strip()
        print(f"WARNING coverage {model} {window} = {coverage:.4f} is outside [0.70, 0.90]")
    rows.append(
        {
            "eval_window": window,
            "model": model,
            "quantile": "",
            "metric": "coverage",
            "value": coverage,
            "n": n,
            "note": coverage_note,
        }
    )
    crossing = quantile_crossing_rate(preds[0.1], preds[0.5], preds[0.9])
    rows.append(
        {
            "eval_window": window,
            "model": model,
            "quantile": "",
            "metric": "quantile_crossing_rate",
            "value": crossing,
            "n": n,
            "note": note,
        }
    )
    ordered = rearrange(preds)
    rows.append(
        {
            "eval_window": window,
            "model": model,
            "quantile": "",
            "metric": "coverage_rearranged",
            "value": empirical_coverage(y, ordered[0.1], ordered[0.9]),
            "n": n,
            "note": "diagnostic only; headline coverage uses the emitted quantiles",
        }
    )
    return rows


def _compare_row(
    model: str,
    window: str,
    metric: str,
    value: float,
    n: int,
    note: str,
) -> dict:
    return {
        "eval_window": window,
        "model": model,
        "quantile": "",
        "metric": metric,
        "value": value,
        "n": n,
        "note": note,
    }


def _run_window(series: pl.DataFrame, cfg, origin_start: datetime, origin_end: datetime) -> dict[str, pl.DataFrame]:
    common = dict(
        embargo_hours=int(cfg.embargo_hours),
        origin_step_hours=int(cfg.origin_step_days) * 24,
        horizon_hours=int(cfg.horizon_hours),
        min_train_hours=int(cfg.min_train_hours),
        origin_start=origin_start,
        origin_end=origin_end,
        time_col=TIME_COL,
        value_col=VALUE_COL,
    )
    models = {
        "seasonal_naive": _naive_fn(cfg),
        "lightgbm": _lightgbm_fn(cfg),
        "ensemble": _ensemble_fn(cfg),
    }
    out: dict[str, pl.DataFrame] = {}
    for name, fn in models.items():
        folds, frame = rolling_origin_cv(series, fn, **common)
        print(f"{name}: origins={len(folds)} rows={frame.height}")
        out[name] = frame
    return out


def _empty_window_row(window: str, note: str) -> dict:
    return {
        "eval_window": window,
        "model": "",
        "quantile": "",
        "metric": "n_origins",
        "value": 0,
        "n": 0,
        "note": note,
    }


def main() -> int:
    sessions_path = REPO_ROOT / "data" / "interim" / "sessions.parquet"
    if not sessions_path.exists():
        print(f"missing {sessions_path}", file=sys.stderr)
        return 1
    cfg_path = REPO_ROOT / "configs" / "forecast.yaml"
    cfg = load_config(cfg_path)
    # Stamp before any artifact is written, or git_sha picks up this run's own output.
    hazard_prov = provenance(REPO_ROOT / "configs" / "site_caltech.yaml")
    forecast_prov = provenance(cfg_path)
    try:
        chronos_zero_shot()
    except ImportError as exc:
        print(f"chronos: {exc}")

    sessions = pl.read_parquet(sessions_path)
    hourly = build_hourly_demand(sessions, "caltech")
    hazards = write_hazard_metrics(sessions, hourly, prov=hazard_prov)
    print_hazard_rows(hazards)

    times = list(hourly[TIME_COL].to_list())
    if not times:
        print("hourly series is empty", file=sys.stderr)
        return 1
    quantiles = [float(tau) for tau in cfg.quantiles]
    mirror_note = (
        f"mirror span {times[0].isoformat()} to {times[-1].isoformat()}; "
        "embargo 336h hides every configured lag on the horizon"
    )
    print(f"hourly hours={hourly.height} {mirror_note}")
    predictions = _run_window(series=hourly, cfg=cfg, origin_start=times[0], origin_end=series_end_exclusive(times))
    if predictions["seasonal_naive"].height == 0:
        print("no mirror origins", file=sys.stderr)
        return 1

    rows: list[dict] = []
    window = "mirror_2018"
    losses: dict[str, np.ndarray] = {}
    for name, frame in predictions.items():
        rows.extend(_score_rows(frame, name, window, quantiles, mirror_note))
        losses[name] = _mean_loss(frame, quantiles)
        rows.append(
            _compare_row(name, window, "n_origins", frame["origin"].n_unique(), frame.height, mirror_note)
        )

    block = int(cfg.bootstrap_block_hours)
    draws = int(cfg.bootstrap_replicates)
    seed = int(cfg.bootstrap_seed)
    naive_loss = losses["seasonal_naive"]
    naive_mean = float(np.nanmean(naive_loss))
    single_means = {
        name: float(np.nanmean(loss)) for name, loss in losses.items() if name != "ensemble"
    }
    best_single = min(single_means, key=single_means.get)
    for name, loss in losses.items():
        mean_pinball = float(np.nanmean(loss))
        rows.append(
            _compare_row(
                name,
                window,
                "mean_pinball",
                mean_pinball,
                int(np.isfinite(loss).sum()),
                mirror_note,
            )
        )
        if name == "seasonal_naive" or naive_mean == 0:
            improvement = 0.0 if name == "seasonal_naive" else float("nan")
        else:
            improvement = (naive_mean - mean_pinball) / naive_mean
        if name != "seasonal_naive":
            test = paired_moving_block_bootstrap(loss, naive_loss, block=block, B=draws, seed=seed)
            rows.append(
                _compare_row(name, window, "pvalue_vs_naive", float(test["p_value"]), int(test["n"]), mirror_note)
            )
            rows.append(
                _compare_row(
                    name,
                    window,
                    "pinball_diff_ci_low",
                    float(test["ci_low"]),
                    int(test["n"]),
                    "95% CI for mean(pinball_model - pinball_naive); negative means the model is lower",
                )
            )
            rows.append(
                _compare_row(
                    name,
                    window,
                    "pinball_diff_ci_high",
                    float(test["ci_high"]),
                    int(test["n"]),
                    "95% CI for mean(pinball_model - pinball_naive); negative means the model is lower",
                )
            )
            rows.append(
                _compare_row(
                    name,
                    window,
                    "pinball_improvement_vs_naive",
                    improvement,
                    int(test["n"]),
                    "(naive_mean - model_mean) / naive_mean; negative means the model is worse",
                )
            )
            print(
                f"{name} mean_pinball={mean_pinball:.6g} vs naive {naive_mean:.6g} "
                f"improvement={improvement:.4%} p={float(test['p_value']):.4g} "
                f"CI=[{float(test['ci_low']):.6g}, {float(test['ci_high']):.6g}]"
            )
        else:
            print(f"seasonal_naive mean_pinball={mean_pinball:.6g}")

    ensemble_vs_best = paired_moving_block_bootstrap(
        losses["ensemble"], losses[best_single], block=block, B=draws, seed=seed
    )
    rows.append(
        _compare_row(
            "ensemble",
            window,
            "pvalue_vs_best_single",
            float(ensemble_vs_best["p_value"]),
            int(ensemble_vs_best["n"]),
            f"best single is {best_single}",
        )
    )
    print(
        f"ensemble vs {best_single}: p={float(ensemble_vs_best['p_value']):.4g} "
        f"mean_diff={float(ensemble_vs_best['mean_diff']):.6g} "
        f"CI=[{float(ensemble_vs_best['ci_low']):.6g}, {float(ensemble_vs_best['ci_high']):.6g}]"
    )

    end_local = times[-1].isoformat()
    rows.append(
        _empty_window_row(
            "primary_2019_06_2020_03",
            f"pre-registered primary window has no origins; mirror ends {end_local}",
        )
    )
    rows.append(
        _empty_window_row(
            "regime_2020_03",
            f"COVID regime window has no origins; mirror ends {end_local}",
        )
    )
    primary_start = localize_origin(cfg.primary_origin_start, times[0])
    primary_end = localize_origin(cfg.primary_origin_end, times[0])
    regime_start = localize_origin(cfg.regime_origin_start, times[0])
    regime_end = localize_origin(cfg.regime_origin_end, times[0])
    for label, start, end in (
        ("primary_2019_06_2020_03", primary_start, primary_end),
        ("regime_2020_03", regime_start, regime_end),
    ):
        folds = iter_folds(
            times,
            embargo_hours=int(cfg.embargo_hours),
            origin_step_hours=int(cfg.origin_step_days) * 24,
            horizon_hours=int(cfg.horizon_hours),
            min_train_hours=int(cfg.min_train_hours),
            origin_start=start,
            origin_end=end,
        )
        print(f"{label}: origins={len(folds)}")
        if folds:
            print(f"WARNING {label} was expected to be empty on this mirror and is not")

    prov = forecast_prov
    for row in rows:
        row.update(prov)
    dest = REPO_ROOT / "results" / "forecast_scores.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "eval_window",
        "model",
        "quantile",
        "metric",
        "value",
        "n",
        "note",
        *prov.keys(),
    ]
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {dest}")

    scores = pl.read_csv(dest)
    choice = pick_forecast(scores, "mirror_2018")
    print(
        f"champion={choice['champion']} mean_pinball={choice['mean_pinball']} "
        f"beats_naive={choice['beats_naive']} p_value_vs_naive={choice['p_value_vs_naive']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
