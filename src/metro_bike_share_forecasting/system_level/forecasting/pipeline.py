from __future__ import annotations

from pathlib import Path

import pandas as pd

from metro_bike_share_forecasting.system_level.forecasting.backtesting import run_backtests
from metro_bike_share_forecasting.system_level.forecasting.config import SystemLevelConfig
from metro_bike_share_forecasting.system_level.forecasting.data import (
    ensure_output_directories,
    load_external_features,
    load_system_level_target,
    write_dataframe,
    write_json,
)
from metro_bike_share_forecasting.system_level.forecasting.evaluation import (
    build_fit_diagnostics_table,
    build_recommendation_table,
    plot_model_comparison,
    plot_production_forecasts,
    summarize_backtest_metrics,
    write_interval_summary_report,
    write_sarimax_review,
    write_system_level_summary,
)
from metro_bike_share_forecasting.system_level.forecasting.features import build_system_level_features
from metro_bike_share_forecasting.system_level.forecasting.intervals import (
    apply_calibrated_intervals,
    build_interval_sample_table,
    collect_backtest_residuals,
    evaluate_interval_quality,
    fit_interval_calibration,
)
from metro_bike_share_forecasting.system_level.forecasting.models import MODEL_REGISTRY


def build_time_index_and_target_artifact(config: SystemLevelConfig) -> pd.DataFrame:
    return load_system_level_target(config)


def build_external_feature_artifact(config: SystemLevelConfig) -> pd.DataFrame:
    return load_external_features(config)


def build_feature_artifact(
    target_frame: pd.DataFrame,
    external_features: pd.DataFrame,
    config: SystemLevelConfig,
) -> pd.DataFrame:
    return build_system_level_features(target_frame, config, external_features)


def run_family_training(
    target_frame: pd.DataFrame,
    external_features: pd.DataFrame,
    config: SystemLevelConfig,
    family: str,
) -> pd.DataFrame:
    if family == "baselines":
        model_keys = [name for name, enabled in config.baselines_enabled.items() if enabled]
    elif family == "classical":
        model_keys = [name for name, enabled in config.classical_enabled.items() if enabled]
    elif family == "ml":
        model_keys = [name for name, enabled in config.ml_enabled.items() if enabled]
    else:
        raise ValueError(f"Unknown family: {family}")

    forecasts: list[pd.DataFrame] = []
    for model_key in model_keys:
        for horizon in config.production_horizons:
            frame = MODEL_REGISTRY[model_key](target_frame, horizon, config, external_features).copy()
            frame["horizon"] = horizon
            frame["model_family"] = family
            forecasts.append(frame)
    return pd.concat(forecasts, ignore_index=True) if forecasts else pd.DataFrame()


def run_system_level_pipeline(config: SystemLevelConfig) -> dict[str, object]:
    directories = ensure_output_directories(config)

    target_frame = build_time_index_and_target_artifact(config)
    external_features = build_external_feature_artifact(config)
    feature_frame = build_feature_artifact(target_frame, external_features, config)

    write_dataframe(target_frame, directories["feature_artifacts"] / "system_level_target.csv")
    write_dataframe(external_features, directories["feature_artifacts"] / "system_level_external_features.csv")
    write_dataframe(feature_frame, directories["feature_artifacts"] / "system_level_features.csv")

    model_keys = config.enabled_model_keys
    backtest_metrics, backtest_forecasts, backtest_windows = run_backtests(target_frame, external_features, config, model_keys)
    summary = summarize_backtest_metrics(backtest_metrics)
    recommendation_table = build_recommendation_table(summary)
    backtest_residuals = collect_backtest_residuals(backtest_forecasts)
    interval_calibration = fit_interval_calibration(backtest_residuals)
    backtest_forecasts = apply_calibrated_intervals(backtest_forecasts, interval_calibration)
    interval_summary, interval_summary_by_step = evaluate_interval_quality(backtest_forecasts)
    backtest_fit_diagnostics = build_fit_diagnostics_table(backtest_forecasts)

    write_dataframe(backtest_metrics, directories["backtests"] / "system_level_fold_metrics.csv")
    write_dataframe(backtest_residuals, directories["backtests"] / "system_level_backtest_residuals.csv")
    write_dataframe(backtest_forecasts, directories["backtests"] / "system_level_fold_forecasts.csv")
    write_dataframe(backtest_windows, directories["backtests"] / "system_level_backtest_windows.csv")
    write_dataframe(backtest_fit_diagnostics, directories["backtests"] / "system_level_fit_diagnostics.csv")
    write_dataframe(summary, directories["metrics"] / "system_level_model_comparison.csv")
    write_dataframe(recommendation_table, directories["metrics"] / "system_level_recommendations.csv")
    write_dataframe(interval_calibration, directories["metrics"] / "system_level_interval_calibration.csv")
    write_dataframe(interval_summary, directories["metrics"] / "system_level_interval_coverage.csv")
    write_dataframe(interval_summary_by_step, directories["metrics"] / "system_level_interval_coverage_by_step.csv")

    baseline_forecasts = run_family_training(target_frame, external_features, config, "baselines")
    classical_forecasts = run_family_training(target_frame, external_features, config, "classical")
    ml_forecasts = run_family_training(target_frame, external_features, config, "ml")
    production_forecasts = pd.concat(
        [frame for frame in (baseline_forecasts, classical_forecasts, ml_forecasts) if not frame.empty],
        ignore_index=True,
    )
    production_forecasts = apply_calibrated_intervals(production_forecasts, interval_calibration)
    write_dataframe(production_forecasts, directories["forecasts"] / "system_level_future_forecasts.csv")
    production_fit_diagnostics = build_fit_diagnostics_table(production_forecasts)
    write_dataframe(production_fit_diagnostics, directories["models"] / "system_level_production_fit_diagnostics.csv")
    interval_sample = build_interval_sample_table(
        production_forecasts,
        preferred_models={7: "ets", 30: "fourier_dynamic_regression"},
    )
    write_dataframe(interval_sample, directories["forecasts"] / "system_level_interval_sample.csv")

    model_registry = pd.DataFrame(
        [
            {"model_name": model_name, "scope": "system_level", "family": family}
            for family, mapping in (
                ("baseline", config.baselines_enabled),
                ("classical", config.classical_enabled),
                ("ml", config.ml_enabled),
            )
            for model_name, enabled in mapping.items()
            if enabled
        ]
    )
    write_dataframe(model_registry, directories["models"] / "system_level_model_registry.csv")
    write_json(
        {
            "scope": "system_level",
            "enabled_models": config.enabled_model_keys,
            "forecast_horizons": list(config.forecast_horizons),
            "extended_horizon": config.extended_horizon,
        },
        directories["models"] / "system_level_run_manifest.json",
    )

    comparison_plot = plot_model_comparison(summary, directories["figures"] / "system_level_model_comparison.png")
    forecast_plot = plot_production_forecasts(production_forecasts, directories["figures"] / "system_level_future_forecasts.png")
    report_path = write_system_level_summary(config, recommendation_table, summary, directories["reports"] / "system_level_summary.md")
    sarimax_review_path = write_sarimax_review(summary, backtest_fit_diagnostics, directories["reports"] / "sarimax_review.md")
    interval_report_path = write_interval_summary_report(
        config,
        summary,
        interval_summary,
        interval_sample,
        directories["reports"] / "system_level_interval_summary.md",
    )

    return {
        "target_rows": len(target_frame),
        "feature_rows": len(feature_frame),
        "backtest_metric_rows": len(backtest_metrics),
        "forecast_rows": len(production_forecasts),
        "output_root": str(config.output_root),
        "report_path": str(report_path),
        "sarimax_review_path": str(sarimax_review_path),
        "interval_report_path": str(interval_report_path),
        "comparison_plot": str(comparison_plot) if comparison_plot else None,
        "forecast_plot": str(forecast_plot) if forecast_plot else None,
    }
