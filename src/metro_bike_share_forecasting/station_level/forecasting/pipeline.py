from __future__ import annotations

from dataclasses import replace

import pandas as pd

from metro_bike_share_forecasting.station_level.forecasting.backtesting import (
    build_station_backtest_windows,
    expand_horizon_rows,
    summarize_fold_metrics,
)
from metro_bike_share_forecasting.station_level.forecasting.config import StationLevelForecastConfig
from metro_bike_share_forecasting.station_level.forecasting.data import (
    active_station_ids_for_production,
    ensure_output_directories,
    load_station_forecast_panel,
    load_station_slice_lookup,
    observed_station_daily,
    write_dataframe,
    write_json,
)
from metro_bike_share_forecasting.station_level.forecasting.evaluation import (
    build_slice_metrics,
    plot_model_comparison,
    summarize_backtest_metrics,
)
from metro_bike_share_forecasting.station_level.forecasting.features import build_station_feature_frame, training_rows
from metro_bike_share_forecasting.station_level.forecasting.models import (
    MODEL_DIAGNOSTIC_COLUMNS,
    fit_deepar_model,
    fit_tree_model,
    predict_deepar,
    predict_naive,
    predict_seasonal_naive_7,
    predict_with_tree,
)
from metro_bike_share_forecasting.system_level.forecasting.intervals import (
    apply_calibrated_intervals,
    collect_backtest_residuals,
    evaluate_interval_quality,
    fit_interval_calibration,
)


MODEL_REGISTRY = {
    "naive": ("baseline", predict_naive),
    "seasonal_naive_7": ("baseline", predict_seasonal_naive_7),
}


def _selected_model_keys(config: StationLevelForecastConfig, model: str) -> list[str]:
    model = model.lower()
    if model == "all":
        return config.enabled_model_keys
    if model == "baseline":
        return [name for name, enabled in config.baselines_enabled.items() if enabled]
    if model in {"lgbm", "xgboost", "deepar"}:
        return [model]
    raise ValueError(f"Unsupported station model selection: {model}")


def _attach_slice_lookup(frame: pd.DataFrame, slice_lookup: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or slice_lookup.empty:
        return frame.copy()
    return frame.merge(slice_lookup, on="station_id", how="left")


def _fill_intervals_with_calibration(forecasts: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    if forecasts.empty:
        return forecasts.copy()
    direct = forecasts.copy()
    calibrated = apply_calibrated_intervals(direct, calibration)
    for column in ["lower_80", "upper_80", "lower_95", "upper_95"]:
        if column in direct.columns:
            calibrated[column] = direct[column].combine_first(calibrated[column])
    return calibrated


def _fit_station_model(
    model_name: str,
    train_panel: pd.DataFrame,
    feature_frame: pd.DataFrame,
    config: StationLevelForecastConfig,
    tune: bool,
) -> tuple[object, pd.DataFrame]:
    if model_name in {"naive", "seasonal_naive_7"}:
        return model_name, pd.DataFrame()
    if model_name in {"lgbm", "xgboost"}:
        return fit_tree_model(feature_frame, config, model_name, tune=tune)
    if model_name == "deepar":
        return fit_deepar_model(feature_frame, config, tune=tune)
    raise ValueError(f"Unsupported station model: {model_name}")


def _predict_station_model(
    model_name: str,
    fitted: object,
    train_panel: pd.DataFrame,
    forecast_dates: pd.DatetimeIndex,
    station_ids: list[str],
    config: StationLevelForecastConfig,
    slice_lookup: pd.DataFrame,
) -> pd.DataFrame:
    if model_name == "naive":
        return predict_naive(train_panel, forecast_dates, station_ids, config)
    if model_name == "seasonal_naive_7":
        return predict_seasonal_naive_7(train_panel, forecast_dates, station_ids, config)
    if model_name in {"lgbm", "xgboost"}:
        return predict_with_tree(fitted, train_panel, forecast_dates, station_ids, config, slice_lookup)
    if model_name == "deepar":
        return predict_deepar(fitted, train_panel, forecast_dates, station_ids, config, slice_lookup)
    raise ValueError(f"Unsupported station model: {model_name}")


def _build_scored_forecasts(
    raw_forecasts: pd.DataFrame,
    actuals: pd.DataFrame,
    train_panel: pd.DataFrame,
    config: StationLevelForecastConfig,
) -> pd.DataFrame:
    if raw_forecasts.empty:
        return raw_forecasts.copy()
    scored = raw_forecasts.merge(actuals.rename(columns={"target": "actual"}), on=["station_id", "date"], how="inner")
    if scored.empty:
        return scored
    scored["horizon_step"] = (pd.to_datetime(scored["date"]) - pd.to_datetime(scored["train_end"])) .dt.days.astype(int)
    return expand_horizon_rows(scored, train_panel, config)


def _production_family(frame: pd.DataFrame, model_name: str) -> str:
    if model_name in {"naive", "seasonal_naive_7"}:
        return "baseline"
    if model_name in {"lgbm", "xgboost"}:
        return "tree"
    return "deep"


def run_station_level_pipeline(
    config: StationLevelForecastConfig,
    model: str = "all",
    tune: bool | None = None,
) -> dict[str, object]:
    directories = ensure_output_directories(config)
    panel = load_station_forecast_panel(config)
    slice_lookup = load_station_slice_lookup(config, panel)
    feature_frame = build_station_feature_frame(panel, config, slice_lookup)
    observed_daily = observed_station_daily(panel)

    write_dataframe(observed_daily, directories["feature_artifacts"] / "station_level_observed_panel.csv")
    write_dataframe(slice_lookup, directories["feature_artifacts"] / "station_level_slice_lookup.csv")

    model_keys = _selected_model_keys(config, model)
    tuning_enabled = config.tune_enabled if tune is None else bool(tune)
    windows = build_station_backtest_windows(panel, config)

    tuning_tables: list[pd.DataFrame] = []
    scored_rows: list[pd.DataFrame] = []
    window_rows: list[dict[str, object]] = []

    for window in windows:
        train_panel = panel.loc[panel["date"] <= window.train_end].copy()
        feature_train = feature_frame.loc[feature_frame["date"] <= window.train_end].copy()
        test_actuals = panel.loc[
            panel["in_service"] & panel["date"].between(window.train_end + pd.Timedelta(days=1), window.test_end),
            ["station_id", "date", "target"],
        ].copy()
        forecast_station_ids = sorted(test_actuals["station_id"].astype(str).unique().tolist())
        if not forecast_station_ids:
            continue
        window_rows.append(
            {
                "fold_id": window.fold_id,
                "train_end": window.train_end,
                "test_end": window.test_end,
                "forecast_station_count": len(forecast_station_ids),
            }
        )

        for model_name in model_keys:
            fitted, tuning_table = _fit_station_model(model_name, train_panel, feature_train, config, tune=tuning_enabled)
            if not tuning_table.empty:
                tuning_table = tuning_table.copy()
                tuning_table["fold_id"] = window.fold_id
                tuning_tables.append(tuning_table)
            raw = _predict_station_model(model_name, fitted, train_panel, window.forecast_dates, forecast_station_ids, config, slice_lookup)
            raw["fold_id"] = window.fold_id
            raw["train_end"] = window.train_end
            for column in MODEL_DIAGNOSTIC_COLUMNS:
                if column not in raw.columns:
                    raw[column] = None
            if hasattr(fitted, "selected_params") and "selected_params" in raw.columns:
                raw["selected_params"] = str(getattr(fitted, "selected_params"))
            if hasattr(fitted, "tuned") and "tuned" in raw.columns:
                raw["tuned"] = bool(getattr(fitted, "tuned"))
            scored_rows.append(_build_scored_forecasts(raw, test_actuals, train_panel, config))

    backtest_forecasts = pd.concat([frame for frame in scored_rows if not frame.empty], ignore_index=True) if scored_rows else pd.DataFrame()
    backtest_forecasts = _attach_slice_lookup(backtest_forecasts, slice_lookup)
    backtest_metrics = summarize_fold_metrics(backtest_forecasts)
    backtest_summary = summarize_backtest_metrics(backtest_metrics)
    backtest_residuals = collect_backtest_residuals(backtest_forecasts)
    calibration = fit_interval_calibration(backtest_residuals)
    backtest_forecasts = _fill_intervals_with_calibration(backtest_forecasts, calibration)
    interval_summary, interval_summary_by_step = evaluate_interval_quality(backtest_forecasts)
    backtest_forecasts = _attach_slice_lookup(backtest_forecasts, slice_lookup)
    if {"lower_80", "upper_80", "actual"}.issubset(backtest_forecasts.columns):
        backtest_forecasts["covered_80"] = (
            (backtest_forecasts["actual"] >= backtest_forecasts["lower_80"])
            & (backtest_forecasts["actual"] <= backtest_forecasts["upper_80"])
        ).astype(float)
        backtest_forecasts["width_80"] = backtest_forecasts["upper_80"] - backtest_forecasts["lower_80"]
    slice_metrics = build_slice_metrics(backtest_forecasts)

    write_dataframe(backtest_forecasts, directories["backtests"] / "station_level_fold_forecasts.csv")
    write_dataframe(backtest_metrics, directories["backtests"] / "station_level_fold_metrics.csv")
    write_dataframe(pd.DataFrame(window_rows), directories["backtests"] / "station_level_backtest_windows.csv")
    write_dataframe(backtest_residuals, directories["backtests"] / "station_level_backtest_residuals.csv")
    write_dataframe(backtest_summary, directories["metrics"] / "station_level_model_comparison.csv")
    write_dataframe(slice_metrics, directories["metrics"] / "station_level_slice_metrics.csv")
    write_dataframe(interval_summary, directories["metrics"] / "station_level_interval_coverage.csv")
    write_dataframe(interval_summary_by_step, directories["metrics"] / "station_level_interval_coverage_by_step.csv")
    write_dataframe(calibration, directories["metrics"] / "station_level_interval_calibration.csv")

    production_station_ids = active_station_ids_for_production(panel, config)
    production_forecasts: list[pd.DataFrame] = []
    production_tuning_tables: list[pd.DataFrame] = []
    if production_station_ids:
        forecast_dates = pd.date_range(panel["date"].max() + pd.Timedelta(days=1), periods=max(config.production_horizons), freq="D")
        for model_name in model_keys:
            fitted, tuning_table = _fit_station_model(model_name, panel, feature_frame, config, tune=tuning_enabled)
            if not tuning_table.empty:
                tuning_table = tuning_table.copy()
                tuning_table["fold_id"] = 0
                production_tuning_tables.append(tuning_table)
            raw = _predict_station_model(model_name, fitted, panel, forecast_dates, production_station_ids, config, slice_lookup)
            for horizon in config.production_horizons:
                subset = raw.loc[(pd.to_datetime(raw["date"]) - pd.to_datetime(panel["date"].max())).dt.days <= horizon].copy()
                subset["horizon"] = horizon
                subset["horizon_step"] = (pd.to_datetime(subset["date"]) - pd.to_datetime(panel["date"].max())).dt.days.astype(int)
                subset["model_family"] = _production_family(subset, model_name)
                production_forecasts.append(subset)

    future_forecasts = pd.concat(production_forecasts, ignore_index=True) if production_forecasts else pd.DataFrame()
    future_forecasts = _attach_slice_lookup(future_forecasts, slice_lookup)
    future_forecasts = _fill_intervals_with_calibration(future_forecasts, calibration)
    write_dataframe(future_forecasts, directories["forecasts"] / "station_level_future_forecasts.csv")

    tuning_results = pd.concat(tuning_tables + production_tuning_tables, ignore_index=True) if tuning_tables or production_tuning_tables else pd.DataFrame()
    if not tuning_results.empty:
        write_dataframe(tuning_results, directories["models"] / "station_level_tuning_results.csv")

    model_registry = pd.DataFrame(
        [
            {"model_name": model_name, "scope": "station_level", "family": _production_family(pd.DataFrame(), model_name)}
            for model_name in model_keys
        ]
    )
    write_dataframe(model_registry, directories["models"] / "station_level_model_registry.csv")
    write_json(
        {
            "scope": "station_level",
            "enabled_models": model_keys,
            "forecast_horizons": list(config.forecast_horizons),
            "extended_horizon": config.extended_horizon,
            "tuned": tuning_enabled,
        },
        directories["models"] / "station_level_run_manifest.json",
    )

    comparison_plot = plot_model_comparison(backtest_summary, directories["figures"] / "station_level_model_comparison.png")
    return {
        "panel_rows": len(panel),
        "observed_rows": len(observed_daily),
        "backtest_metric_rows": len(backtest_metrics),
        "forecast_rows": len(future_forecasts),
        "output_root": str(config.output_root),
        "comparison_plot": str(comparison_plot) if comparison_plot else None,
    }
