from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from system_level.common.cli_utils import (
    default_forecast_package_report,
    noop_progress,
    runtime_environment_notes,
    runtime_environment_report,
)
from system_level.common.intervals import (
    apply_calibrated_intervals,
    collect_backtest_residuals,
    evaluate_interval_quality,
    fit_interval_calibration,
)
from system_level.forecasting.backtesting import run_backtests
from system_level.forecasting.config import SystemLevelConfig
from system_level.forecasting.data import (
    ensure_output_directories,
    load_external_features,
    load_system_level_target,
    write_dataframe,
    write_json,
    write_text,
)
from system_level.forecasting.evaluation import (
    build_recommendation_table,
    build_fit_diagnostics_table,
    plot_model_comparison,
    plot_production_forecasts,
    summarize_backtest_metrics,
)
from system_level.forecasting.features import build_system_level_features
from system_level.forecasting.models import MODEL_REGISTRY
from system_level.forecasting.models import system_model_runtime_notes, system_model_runtime_report


def _system_runtime_metadata(config: SystemLevelConfig) -> dict[str, object]:
    return {
        "environment": runtime_environment_report(),
        "packages": pd.DataFrame(default_forecast_package_report()),
        "notes": runtime_environment_notes() + system_model_runtime_notes(config),
        "configured_models": system_model_runtime_report(config),
    }


def write_system_level_runtime_outputs(
    directories: dict[str, Path],
    runtime_metadata: dict[str, object],
) -> None:
    write_json(runtime_metadata["environment"], directories["models"] / "system_level_runtime_environment.json")
    write_dataframe(runtime_metadata["packages"], directories["models"] / "system_level_package_report.csv")
    write_dataframe(runtime_metadata["configured_models"], directories["models"] / "system_level_configured_model_status.csv")
    write_text("\n".join(runtime_metadata["notes"]) + "\n", directories["models"] / "system_level_runtime_notes.txt")


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


def run_system_level_backtest_stage(
    target_frame: pd.DataFrame,
    external_features: pd.DataFrame,
    config: SystemLevelConfig,
    progress: Callable[[str], None] | None = None,
) -> dict[str, pd.DataFrame]:
    progress = progress or noop_progress
    model_keys = config.enabled_model_keys
    progress(f"Starting system-level backtests for {len(model_keys)} model(s).")
    backtest_metrics, backtest_forecasts, backtest_windows = run_backtests(
        target_frame,
        external_features,
        config,
        model_keys,
        progress=progress,
    )
    summary = summarize_backtest_metrics(backtest_metrics)
    recommendations = build_recommendation_table(summary)
    backtest_residuals = collect_backtest_residuals(backtest_forecasts)
    interval_calibration = fit_interval_calibration(backtest_residuals)
    backtest_forecasts = apply_calibrated_intervals(backtest_forecasts, interval_calibration)
    interval_summary, interval_summary_by_step = evaluate_interval_quality(backtest_forecasts)
    backtest_fit_diagnostics = build_fit_diagnostics_table(backtest_forecasts)
    progress("System-level backtest stage complete.")
    return {
        "backtest_metrics": backtest_metrics,
        "backtest_forecasts": backtest_forecasts,
        "backtest_windows": backtest_windows,
        "backtest_residuals": backtest_residuals,
        "summary": summary,
        "recommendations": recommendations,
        "interval_calibration": interval_calibration,
        "interval_summary": interval_summary,
        "interval_summary_by_step": interval_summary_by_step,
        "backtest_fit_diagnostics": backtest_fit_diagnostics,
    }


def write_system_level_backtest_outputs(directories: dict[str, Path], stage_outputs: dict[str, pd.DataFrame]) -> None:
    write_dataframe(stage_outputs["backtest_metrics"], directories["backtests"] / "system_level_fold_metrics.csv")
    write_dataframe(stage_outputs["backtest_residuals"], directories["backtests"] / "system_level_backtest_residuals.csv")
    write_dataframe(stage_outputs["backtest_forecasts"], directories["backtests"] / "system_level_fold_forecasts.csv")
    write_dataframe(stage_outputs["backtest_windows"], directories["backtests"] / "system_level_backtest_windows.csv")
    write_dataframe(stage_outputs["backtest_fit_diagnostics"], directories["backtests"] / "system_level_fit_diagnostics.csv")
    write_dataframe(stage_outputs["summary"], directories["metrics"] / "system_level_model_comparison.csv")
    write_dataframe(stage_outputs["recommendations"], directories["metrics"] / "system_level_recommended_models.csv")
    write_dataframe(stage_outputs["interval_calibration"], directories["metrics"] / "system_level_interval_calibration.csv")
    write_dataframe(stage_outputs["interval_summary"], directories["metrics"] / "system_level_interval_coverage.csv")
    write_dataframe(stage_outputs["interval_summary_by_step"], directories["metrics"] / "system_level_interval_coverage_by_step.csv")


def run_system_level_production_stage(
    target_frame: pd.DataFrame,
    external_features: pd.DataFrame,
    config: SystemLevelConfig,
    interval_calibration: pd.DataFrame,
    family: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, pd.DataFrame]:
    progress = progress or noop_progress
    if family is None:
        families = ("baselines", "classical", "ml")
    else:
        families = (family,)
    progress(f"Starting production forecasting for families: {', '.join(families)}.")

    family_forecasts = [
        run_family_training(target_frame, external_features, config, selected_family)
        for selected_family in families
    ]
    production_forecasts = pd.concat([frame for frame in family_forecasts if not frame.empty], ignore_index=True)
    production_forecasts = apply_calibrated_intervals(production_forecasts, interval_calibration)
    production_fit_diagnostics = build_fit_diagnostics_table(production_forecasts)
    if production_forecasts.empty:
        model_registry = system_model_runtime_report(config).assign(scope="system_level")
    else:
        runtime_report = system_model_runtime_report(config).rename(columns={"model_name": "report_model_name"})
        model_registry = (
            production_forecasts[["model_name", "model_family"]]
            .drop_duplicates()
            .rename(columns={"model_family": "family"})
            .assign(
                report_model_name=lambda frame: frame["model_name"].replace(
                    {
                        "lightgbm": "tree_boosting",
                        "xgboost": "tree_boosting",
                        "tree_boosting_fallback": "tree_boosting",
                    }
                )
            )
            .assign(scope="system_level")
            .merge(
                runtime_report[["report_model_name", "implementation", "experimental", "tuning_strategy", "note"]],
                on="report_model_name",
                how="left",
            )
            .drop(columns=["report_model_name"])
        )
    progress("System-level production stage complete.")
    return {
        "production_forecasts": production_forecasts,
        "production_fit_diagnostics": production_fit_diagnostics,
        "model_registry": model_registry,
    }


def write_system_level_production_outputs(
    directories: dict[str, Path],
    stage_outputs: dict[str, pd.DataFrame],
    config: SystemLevelConfig,
    family: str | None = None,
    runtime_metadata: dict[str, object] | None = None,
) -> None:
    suffix = "system_level_future_forecasts.csv" if family is None else f"system_level_{family}_forecasts.csv"
    write_dataframe(stage_outputs["production_forecasts"], directories["forecasts"] / suffix)
    write_dataframe(stage_outputs["production_fit_diagnostics"], directories["models"] / "system_level_production_fit_diagnostics.csv")
    write_dataframe(stage_outputs["model_registry"], directories["models"] / "system_level_model_registry.csv")
    write_json(
        {
            "scope": "system_level",
            "enabled_models": config.enabled_model_keys,
            "forecast_horizons": list(config.forecast_horizons),
            "extended_horizon": config.extended_horizon,
            "mase_season_length": config.mase_season_length,
            "missing_target_strategy": config.missing_target_strategy,
            "family": family or "all",
            "tuning_strategy": "heuristic_classical_and_fixed_ml_defaults",
            "runtime_environment_path": str(directories["models"] / "system_level_runtime_environment.json"),
            "package_report_path": str(directories["models"] / "system_level_package_report.csv"),
            "runtime_notes_path": str(directories["models"] / "system_level_runtime_notes.txt"),
            "configured_model_status_path": str(directories["models"] / "system_level_configured_model_status.csv"),
            "native_tree_backends": {
                row["model_name"]: bool(row["native_backend_available"])
                for row in (
                    [] if runtime_metadata is None else runtime_metadata["configured_models"].to_dict(orient="records")
                )
                if row["model_name"] == "tree_boosting"
            },
        },
        directories["models"] / "system_level_run_manifest.json",
    )


def run_system_level_pipeline(
    config: SystemLevelConfig,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    progress = progress or noop_progress
    directories = ensure_output_directories(config)
    runtime_metadata = _system_runtime_metadata(config)
    progress("Writing runtime diagnostics.")
    write_system_level_runtime_outputs(directories, runtime_metadata)

    progress("Loading target and external features.")
    target_frame = build_time_index_and_target_artifact(config)
    external_features = build_external_feature_artifact(config)
    progress(f"Target rows={len(target_frame)} external_rows={len(external_features)}.")
    progress("Building feature artifact.")
    feature_frame = build_feature_artifact(target_frame, external_features, config)

    progress("Writing feature artifacts.")
    write_dataframe(target_frame, directories["feature_artifacts"] / "system_level_target.csv")
    write_dataframe(external_features, directories["feature_artifacts"] / "system_level_external_features.csv")
    write_dataframe(feature_frame, directories["feature_artifacts"] / "system_level_features.csv")

    backtest_outputs = run_system_level_backtest_stage(target_frame, external_features, config, progress=progress)
    progress("Writing backtest outputs.")
    write_system_level_backtest_outputs(directories, backtest_outputs)

    production_outputs = run_system_level_production_stage(
        target_frame,
        external_features,
        config,
        backtest_outputs["interval_calibration"],
        progress=progress,
    )
    progress("Writing production outputs.")
    write_system_level_production_outputs(directories, production_outputs, config, runtime_metadata=runtime_metadata)

    progress("Rendering forecasting figures.")
    comparison_plot = plot_model_comparison(backtest_outputs["summary"], directories["figures"] / "system_level_model_comparison.png")
    forecast_plot = plot_production_forecasts(
        production_outputs["production_forecasts"],
        directories["figures"] / "system_level_future_forecasts.png",
    )
    progress("System-level forecasting pipeline complete.")

    return {
        "target_rows": len(target_frame),
        "feature_rows": len(feature_frame),
        "backtest_metric_rows": len(backtest_outputs["backtest_metrics"]),
        "forecast_rows": len(production_outputs["production_forecasts"]),
        "output_root": str(config.output_root),
        "comparison_plot": str(comparison_plot) if comparison_plot else None,
        "forecast_plot": str(forecast_plot) if forecast_plot else None,
    }
