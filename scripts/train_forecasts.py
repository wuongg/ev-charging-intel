"""Fit the day-2 forecast stack on the Caltech hourly series and write scores.

Chronos is not a member of this run. chronos-forecasting depends on torch, so
the member is dropped and the ensemble is seasonal-naive plus LightGBM.
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from eci.config import REPO_ROOT, load_config, provenance
from eci.data.features import build_hourly_demand
from eci.data.measures import print_hazard_rows, write_hazard_metrics
from eci.forecast.backtest import (
    iter_folds,
    localize_origin,
    seasonal_naive_causal,
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
        return seasonal_naive_causal(
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
        # No cutoff mask. shift/rolling already use actuals strictly before each row.
        frame = build_feature_frame(
            series,
            lags=lags,
            windows=windows,
            value_col=VALUE_COL,
            time_col=TIME_COL,
            cutoff=None,
        )
        matrix = frame.select(names).to_numpy()
        target = np.asarray(series[VALUE_COL].to_numpy(), dtype=float)
        train_target = target[fold.train_idx]
        train_matrix = matrix[fold.train_idx]
        models = train_lightgbm_quantile(train_matrix, train_target, names, quantiles, fitted)
        return predict_quantiles(models, matrix[fold.test_idx])

    return predict


def history_null_fraction(series: pl.DataFrame, fold, cfg) -> float:
    """Share of lag and rolling cells on the test horizon that are missing."""
    lags = [int(lag) for lag in cfg.lags]
    windows = [int(window) for window in cfg.rolling_windows]
    names = [name for name in feature_column_names(lags, windows) if name.startswith(("lag_", "roll_"))]
    frame = build_feature_frame(
        series,
        lags=lags,
        windows=windows,
        value_col=VALUE_COL,
        time_col=TIME_COL,
        cutoff=None,
    )
    matrix = frame.gather(fold.test_idx.tolist()).select(names).to_numpy()
    if matrix.size == 0:
        return float("nan")
    return float(np.mean(~np.isfinite(matrix)))


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


def _predict_frame(series: pl.DataFrame, fold, predicted: dict[float, np.ndarray]) -> pl.DataFrame:
    test = series.gather(fold.test_idx.tolist())
    columns: dict[str, object] = {
        TIME_COL: test[TIME_COL],
        "y": test[VALUE_COL],
        "origin": [fold.origin] * int(fold.test_idx.size),
    }
    for level, values in predicted.items():
        columns[f"q{int(round(float(level) * 100)):02d}"] = np.asarray(values, dtype=float)
    return pl.DataFrame(columns)


def _run_window(series: pl.DataFrame, cfg, origin_start: datetime, origin_end: datetime):
    """One pass per origin. LightGBM is fit once and reused by the ensemble."""
    folds = iter_folds(
        list(series[TIME_COL].to_list()),
        embargo_hours=int(cfg.embargo_hours),
        origin_step_hours=int(cfg.origin_step_days) * 24,
        horizon_hours=int(cfg.horizon_hours),
        min_train_hours=int(cfg.min_train_hours),
        origin_start=origin_start,
        origin_end=origin_end,
    )
    naive_fn = _naive_fn(cfg)
    lightgbm_fn = _lightgbm_fn(cfg)
    buckets: dict[str, list[pl.DataFrame]] = {name: [] for name in ("seasonal_naive", "lightgbm", "ensemble")}
    for index, fold in enumerate(folds, start=1):
        naive = naive_fn(series, fold)
        lightgbm = lightgbm_fn(series, fold)
        ensemble = rearrange(equal_weight_quantiles([naive, lightgbm]))
        buckets["seasonal_naive"].append(_predict_frame(series, fold, naive))
        buckets["lightgbm"].append(_predict_frame(series, fold, lightgbm))
        buckets["ensemble"].append(_predict_frame(series, fold, ensemble))
        if index == 1 or index == len(folds) or index % 5 == 0:
            print(f"origin {index}/{len(folds)} {fold.origin.isoformat()}")
    out = {name: pl.concat(frames, how="vertical") if frames else pl.DataFrame() for name, frames in buckets.items()}
    print(f"v2 origins={len(folds)} rows={out['seasonal_naive'].height if folds else 0}")
    return out, folds


def _preserved_v1_rows(path: Path) -> list[dict]:
    """Keep the day-2 mirror scores. Their note is marked information-asymmetric."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        loaded = list(csv.DictReader(handle))
    kept = []
    for row in loaded:
        if row.get("eval_window") != "mirror_2018":
            continue
        note = row.get("note") or ""
        if "v1, information-asymmetric" not in note:
            row["note"] = f"v1, information-asymmetric; {note}".strip()
        kept.append(row)
    return kept


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
    v2_note = (
        f"v2, features-from-actuals; span {times[0].isoformat()} to {times[-1].isoformat()}; "
        "embargo 336h restricts training rows only"
    )
    print(f"hourly hours={hourly.height} {v2_note}")
    predictions, v2_folds = _run_window(
        series=hourly, cfg=cfg, origin_start=times[0], origin_end=series_end_exclusive(times)
    )
    if predictions["seasonal_naive"].height == 0:
        print("no extended origins", file=sys.stderr)
        return 1

    rows: list[dict] = []
    window = "extended_2018_2020"
    losses: dict[str, np.ndarray] = {}
    for name, frame in predictions.items():
        rows.extend(_score_rows(frame, name, window, quantiles, v2_note))
        losses[name] = _mean_loss(frame, quantiles)
        rows.append(
            _compare_row(name, window, "n_origins", frame["origin"].n_unique(), frame.height, v2_note)
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
                v2_note,
            )
        )
        if name == "seasonal_naive" or naive_mean == 0:
            improvement = 0.0 if name == "seasonal_naive" else float("nan")
        else:
            improvement = (naive_mean - mean_pinball) / naive_mean
        if name != "seasonal_naive":
            test = paired_moving_block_bootstrap(loss, naive_loss, block=block, B=draws, seed=seed)
            rows.append(
                _compare_row(name, window, "pvalue_vs_naive", float(test["p_value"]), int(test["n"]), v2_note)
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
    primary_start = localize_origin(cfg.primary_origin_start, times[0])
    primary_end = localize_origin(cfg.primary_origin_end, times[0])
    regime_start = localize_origin(cfg.regime_origin_start, times[0])
    regime_end = localize_origin(cfg.regime_origin_end, times[0])
    for label, start, end in (
        ("primary_2019_06_2020_03", primary_start, primary_end),
        ("regime_2020_03", regime_start, regime_end),
    ):
        window_folds = iter_folds(
            times,
            embargo_hours=int(cfg.embargo_hours),
            origin_step_hours=int(cfg.origin_step_days) * 24,
            horizon_hours=int(cfg.horizon_hours),
            min_train_hours=int(cfg.min_train_hours),
            origin_start=start,
            origin_end=end,
        )
        print(f"{label}: origins={len(window_folds)}")
        if not window_folds:
            rows.append(
                _empty_window_row(
                    label,
                    f"no origin landed in this window; series ends {end_local}",
                )
            )
        else:
            rows.append(
                _compare_row(
                    "",
                    label,
                    "n_origins",
                    len(window_folds),
                    len(window_folds) * int(cfg.horizon_hours),
                    "subset of the extended_2018_2020 origins; same fits, not a second training run",
                )
            )

    diagnosis = []
    lightgbm_frame = predictions["lightgbm"]
    for fold in v2_folds:
        sub = lightgbm_frame.filter(pl.col("origin") == fold.origin)
        coverage = empirical_coverage(sub["y"].to_numpy(), sub["q10"].to_numpy(), sub["q90"].to_numpy())
        diagnosis.append(
            {
                "eval_window": window,
                "origin": fold.origin.isoformat(),
                "coverage": coverage,
                "null_feature_fraction": history_null_fraction(hourly, fold, cfg),
                "n": sub.height,
                "note": v2_note,
            }
        )
    outside = [row for row in diagnosis if not (0.70 <= row["coverage"] <= 0.90)]
    if outside:
        print(f"WARNING {len(outside)}/{len(diagnosis)} origins have LightGBM coverage outside [0.70, 0.90]")

    preserved = _preserved_v1_rows(REPO_ROOT / "results" / "forecast_scores.csv")
    prov = forecast_prov
    for row in rows:
        row.update(prov)
    rows = preserved + rows
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
    diagnosis_path = REPO_ROOT / "results" / "forecast_diagnosis.csv"
    with diagnosis_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["eval_window", "origin", "coverage", "null_feature_fraction", "n", "note", *prov.keys()],
        )
        writer.writeheader()
        for row in diagnosis:
            row.update(prov)
            writer.writerow(row)
    print(f"wrote {diagnosis_path}")

    scores = pl.read_csv(dest, infer_schema_length=None)
    choice = pick_forecast(scores, "extended_2018_2020")
    print(
        f"champion={choice['champion']} mean_pinball={choice['mean_pinball']} "
        f"beats_naive={choice['beats_naive']} p_value_vs_naive={choice['p_value_vs_naive']}"
    )
    return 0


def retrain_trigger_demo(
    hourly: pl.DataFrame | None = None,
    *,
    train_end=None,
    score_end=None,
    source: str = "acn_2018_2020_proxy",
) -> dict:
    """Champion LightGBM on the reference years; refit when pinball ratio > 1 for 14 days."""
    from datetime import date

    from eci.data.features import build_hourly_demand
    from eci.eval.drift import run_retrain_demo

    if hourly is None:
        sessions = pl.read_parquet(REPO_ROOT / "data" / "interim" / "sessions.parquet")
        hourly = build_hourly_demand(sessions, "caltech")
    row = run_retrain_demo(
        hourly,
        train_end=train_end or date(2019, 12, 31),
        score_end=score_end,
        source=source,
    )
    dest = REPO_ROOT / "results" / "retrain_demo.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(
        f"retrain trigger_day={row['trigger_day'] or 'none'} "
        f"before={row['before_trigger_pinball']:.4g} after={row['after_refit_pinball']:.4g} "
        f"improvement={row['improvement_pct']:.2f}%"
    )
    return row


if __name__ == "__main__":
    if "--retrain-demo" in sys.argv:
        sys.exit(0 if retrain_trigger_demo() else 1)
    sys.exit(main())
