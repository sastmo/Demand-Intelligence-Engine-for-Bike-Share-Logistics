from __future__ import annotations

from typing import Callable

from dataclasses import replace

import pandas as pd

from system_level.common.intervals import (
    apply_calibrated_intervals,
    collect_backtest_residuals,
    evaluate_interval_quality,
    fit_interval_calibration,
)
from system_level.common.cli_utils import (
    default_forecast_package_report,
    noop_progress,
    runtime_environment_notes,
    runtime_environment_report,
)
from station_level.forecasting.backtesting import (
    build_station_backtest_windows,
    expand_horizon_rows,
    summarize_fold_metrics,
)
from station_level.forecasting.config import StationLevelForecastConfig
from station_level.forecasting.data import (
    active_station_ids_for_production,
    ensure_output_directories,
    load_station_forecast_panel,
    load_station_slice_lookup,
    observed_station_daily,
    write_dataframe,
    write_json,
    write_text,
)
from station_level.forecasting.evaluation import (
    build_recommendation_table,
    build_slice_metrics,
    plot_model_comparison,
    summarize_backtest_metrics,
)
from station_level.forecasting.features import build_station_feature_frame, training_rows
from station_level.forecasting.models import (
    MODEL_DIAGNOSTIC_COLUMNS,
    fit_deepar_model,
    fit_tree_model,
    predict_naive,
    predict_seasonal_naive_7,
    predict_with_deepar,
    predict_with_tree,
    station_model_runtime_notes,
    station_model_runtime_report,
)


MODEL_REGISTRY = {
    "naive": ("baseline", predict_naive),
    "seasonal_naive_7": ("baseline", predict_seasonal_naive_7),
}


def _station_runtime_metadata(config: StationLevelForecastConfig) -> dict[str, object]:
    return {
        "environment": runtime_environment_report(),
        "packages": pd.DataFrame(default_forecast_package_report()),
        "notes": runtime_environment_notes() + station_model_runtime_notes(config),
        "configured_models": station_model_runtime_report(config),
    }


def write_station_level_runtime_outputs(
    directories: dict[str, object],
    runtime_metadata: dict[str, object],
) -> None:
    write_json(runtime_metadata["environment"], directories["models"] / "station_level_runtime_environment.json")
    write_dataframe(runtime_metadata["packages"], directories["models"] / "station_level_package_report.csv")
    write_dataframe(runtime_metadata["configured_models"], directories["models"] / "station_level_configured_model_status.csv")
    write_text("\n".join(runtime_metadata["notes"]) + "\n", directories["models"] / "station_level_runtime_notes.txt")


def _station_model_registry_row(model_name: str, family: str, raw: pd.DataFrame, fitted: object, tune: bool) -> dict[str, object]:
    implementation = ""
    if "implementation" in raw.columns and raw["implementation"].notna().any():
        implementation = str(raw["implementation"].dropna().iloc[0])
    selected_params = ""
    if "selected_params" in raw.columns and raw["selected_params"].notna().any():
        selected_params = str(raw["selected_params"].dropna().iloc[0])
    tuned_value = bool(getattr(fitted, "tuned", tune))
    note = ""
    if implementation in {"hist_gradient_boosting", "gradient_boosting"}:
        note = "Fallback sklearn backend was used because the native tree library was unavailable."
    elif implementation == "global_neural_mlp":
        note = "Experimental neural benchmark; not a canonical recurrent DeepAR implementation."
    elif model_name == "naive":
        note = "Last-observation carry-forward benchmark."
    elif model_name == "seasonal_naive_7":
        note = "Seven-day seasonal naive benchmark."
    return {
        "model_name": model_name,
        "scope": "station_level",
        "family": family,
        "implementation": implementation or model_name,
        "tuned": tuned_value,
        "selected_params": selected_params,
        "experimental": implementation == "global_neural_mlp",
        "note": note,
    }


def _selected_model_keys(config: StationLevelForecastConfig, model: str) -> list[str]:
    model = model.lower()
    if model != "all":
        raise ValueError("Station-level forecasting currently supports only model='all'.")
    return config.enabled_model_keys


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
        return predict_with_deepar(fitted, train_panel, forecast_dates, station_ids, config, slice_lookup)
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


def build_station_level_artifacts(config: StationLevelForecastConfig) -> dict[str, pd.DataFrame]:
    panel = load_station_forecast_panel(config)
    slice_lookup = load_station_slice_lookup(config, panel)
    feature_frame = build_station_feature_frame(panel, config, slice_lookup)
    observed_daily = observed_station_daily(panel)
    return {
        "panel": panel,
        "slice_lookup": slice_lookup,
        "feature_frame": feature_frame,
        "observed_daily": observed_daily,
    }


def write_station_level_feature_artifacts(directories: dict[str, object], artifacts: dict[str, pd.DataFrame]) -> None:
    write_dataframe(artifacts["observed_daily"], directories["feature_artifacts"] / "station_level_observed_panel.csv")
    write_dataframe(artifacts["slice_lookup"], directories["feature_artifacts"] / "station_level_slice_lookup.csv")
    write_dataframe(artifacts["feature_frame"], directories["feature_artifacts"] / "station_level_features.csv")


def run_station_level_backtest_stage(
    config: StationLevelForecastConfig,
    panel: pd.DataFrame,
    slice_lookup: pd.DataFrame,
    feature_frame: pd.DataFrame,
    model: str = "all",
    tune: bool | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, pd.DataFrame | list[str] | bool]:
    progress = progress or noop_progress
    model_keys = _selected_model_keys(config, model)
    tuning_enabled = config.tune_enabled if tune is None else bool(tune)
    windows = build_station_backtest_windows(panel, config)
    progress(
        f"Backtests: {len(windows)} fold(s), {len(model_keys)} model(s), tuning={'on' if tuning_enabled else 'off'}."
    )

    tuning_tables: list[pd.DataFrame] = []
    scored_rows: list[pd.DataFrame] = []
    window_rows: list[dict[str, object]] = []

    for window_index, window in enumerate(windows, start=1):
        train_panel = panel.loc[panel["date"] <= window.train_end].copy()
        feature_train = feature_frame.loc[feature_frame["date"] <= window.train_end].copy()
        test_actuals = panel.loc[
            panel["in_service"] & panel["date"].between(window.train_end + pd.Timedelta(days=1), window.test_end),
            ["station_id", "date", "target"],
        ].copy()
        forecast_station_ids = sorted(test_actuals["station_id"].astype(str).unique().tolist())
        if not forecast_station_ids:
            progress(f"Fold {window_index}/{len(windows)} skipped: no active stations in the forecast window.")
            continue
        progress(
            f"Fold {window_index}/{len(windows)}: train_end={window.train_end.date()} "
            f"test_end={window.test_end.date()} stations={len(forecast_station_ids)}."
        )
        window_rows.append(
            {
                "fold_id": window.fold_id,
                "train_end": window.train_end,
                "test_end": window.test_end,
                "forecast_station_count": len(forecast_station_ids),
            }
        )

        for model_index, model_name in enumerate(model_keys, start=1):
            progress(f"Fold {window_index}/{len(windows)} model {model_index}/{len(model_keys)}: {model_name}.")
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
    recommendations = build_recommendation_table(backtest_summary)
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
    progress(
        f"Backtests complete: metric_rows={len(backtest_metrics)} forecast_rows={len(backtest_forecasts)}."
    )

    return {
        "requested_model": model,
        "tuning_enabled": tuning_enabled,
        "enabled_model_keys": model_keys,
        "backtest_forecasts": backtest_forecasts,
        "backtest_metrics": backtest_metrics,
        "backtest_windows": pd.DataFrame(window_rows),
        "backtest_residuals": backtest_residuals,
        "backtest_summary": backtest_summary,
        "recommendations": recommendations,
        "interval_calibration": calibration,
        "interval_summary": interval_summary,
        "interval_summary_by_step": interval_summary_by_step,
        "slice_metrics": slice_metrics,
        "tuning_results": pd.concat(tuning_tables, ignore_index=True) if tuning_tables else pd.DataFrame(),
    }


def write_station_level_backtest_outputs(directories: dict[str, object], stage_outputs: dict[str, object]) -> None:
    write_dataframe(stage_outputs["backtest_forecasts"], directories["backtests"] / "station_level_fold_forecasts.csv")
    write_dataframe(stage_outputs["backtest_metrics"], directories["backtests"] / "station_level_fold_metrics.csv")
    write_dataframe(stage_outputs["backtest_windows"], directories["backtests"] / "station_level_backtest_windows.csv")
    write_dataframe(stage_outputs["backtest_residuals"], directories["backtests"] / "station_level_backtest_residuals.csv")
    write_dataframe(stage_outputs["backtest_summary"], directories["metrics"] / "station_level_model_comparison.csv")
    write_dataframe(stage_outputs["recommendations"], directories["metrics"] / "station_level_recommended_models.csv")
    write_dataframe(stage_outputs["slice_metrics"], directories["metrics"] / "station_level_slice_metrics.csv")
    write_dataframe(stage_outputs["interval_summary"], directories["metrics"] / "station_level_interval_coverage.csv")
    write_dataframe(stage_outputs["interval_summary_by_step"], directories["metrics"] / "station_level_interval_coverage_by_step.csv")
    write_dataframe(stage_outputs["interval_calibration"], directories["metrics"] / "station_level_interval_calibration.csv")


def run_station_level_production_stage(
    config: StationLevelForecastConfig,
    panel: pd.DataFrame,
    slice_lookup: pd.DataFrame,
    feature_frame: pd.DataFrame,
    interval_calibration: pd.DataFrame,
    model_keys: list[str],
    tune: bool,
    requested_model: str,
    progress: Callable[[str], None] | None = None,
) -> dict[str, pd.DataFrame]:
    progress = progress or noop_progress
    production_station_ids = active_station_ids_for_production(panel, config)
    production_forecasts: list[pd.DataFrame] = []
    production_tuning_tables: list[pd.DataFrame] = []
    registry_rows: list[dict[str, object]] = []
    progress(
        f"Production: {len(production_station_ids)} active station(s), {len(model_keys)} model(s), tune={'on' if tune else 'off'}."
    )

    if production_station_ids:
        forecast_dates = pd.date_range(panel["date"].max() + pd.Timedelta(days=1), periods=max(config.production_horizons), freq="D")
        for model_index, model_name in enumerate(model_keys, start=1):
            progress(f"Production model {model_index}/{len(model_keys)}: {model_name}.")
            fitted, tuning_table = _fit_station_model(model_name, panel, feature_frame, config, tune=tune)
            if not tuning_table.empty:
                tuning_table = tuning_table.copy()
                tuning_table["fold_id"] = 0
                production_tuning_tables.append(tuning_table)
            raw = _predict_station_model(model_name, fitted, panel, forecast_dates, production_station_ids, config, slice_lookup)
            registry_rows.append(_station_model_registry_row(model_name, _production_family(raw, model_name), raw, fitted, tune))
            for horizon in config.production_horizons:
                subset = raw.loc[(pd.to_datetime(raw["date"]) - pd.to_datetime(panel["date"].max())).dt.days <= horizon].copy()
                subset["horizon"] = horizon
                subset["horizon_step"] = (pd.to_datetime(subset["date"]) - pd.to_datetime(panel["date"].max())).dt.days.astype(int)
                subset["model_family"] = _production_family(subset, model_name)
                production_forecasts.append(subset)

    future_forecasts = pd.concat(production_forecasts, ignore_index=True) if production_forecasts else pd.DataFrame()
    future_forecasts = _attach_slice_lookup(future_forecasts, slice_lookup)
    future_forecasts = _fill_intervals_with_calibration(future_forecasts, interval_calibration)
    progress(f"Production complete: forecast_rows={len(future_forecasts)}.")
    tuning_results = pd.concat(production_tuning_tables, ignore_index=True) if production_tuning_tables else pd.DataFrame()
    if registry_rows:
        model_registry = pd.DataFrame(registry_rows).drop_duplicates(subset=["model_name", "implementation"], keep="last")
    else:
        model_registry = station_model_runtime_report(config).assign(scope="station_level", tuned=tune, selected_params="", note=lambda frame: frame["note"].fillna(""))
    manifest = pd.DataFrame(
        [
            {
                "scope": "station_level",
                "requested_model": requested_model,
                "enabled_models": "|".join(model_keys),
                "forecast_horizons": "|".join(str(value) for value in config.forecast_horizons),
                "extended_horizon": config.extended_horizon,
                "tuned": tune,
                "mase_season_length": config.mase_season_length,
                "tuning_strategy": "single_validation_split_small_grid" if tune else "fixed_defaults",
                "experimental_models": "deepar" if config.deepar_enabled else "",
            }
        ]
    )
    return {
        "future_forecasts": future_forecasts,
        "tuning_results": tuning_results,
        "model_registry": model_registry,
        "manifest_table": manifest,
    }


def write_station_level_production_outputs(
    directories: dict[str, object],
    stage_outputs: dict[str, pd.DataFrame],
    config: StationLevelForecastConfig,
    requested_model: str,
    enabled_model_keys: list[str],
    tune: bool,
    runtime_metadata: dict[str, object] | None = None,
) -> None:
    write_dataframe(stage_outputs["future_forecasts"], directories["forecasts"] / "station_level_future_forecasts.csv")
    if not stage_outputs["tuning_results"].empty:
        write_dataframe(stage_outputs["tuning_results"], directories["models"] / "station_level_tuning_results.csv")
    else:
        tuning_path = directories["models"] / "station_level_tuning_results.csv"
        if tuning_path.exists():
            tuning_path.unlink()
    write_dataframe(stage_outputs["model_registry"], directories["models"] / "station_level_model_registry.csv")
    write_json(
        {
            "scope": "station_level",
            "requested_model": requested_model,
            "enabled_models": enabled_model_keys,
            "forecast_horizons": list(config.forecast_horizons),
            "extended_horizon": config.extended_horizon,
            "tuned": tune,
            "mase_season_length": config.mase_season_length,
            "tuning_strategy": "single_validation_split_small_grid" if tune else "fixed_defaults",
            "experimental_models": ["deepar"] if config.deepar_enabled else [],
            "runtime_environment_path": str(directories["models"] / "station_level_runtime_environment.json"),
            "package_report_path": str(directories["models"] / "station_level_package_report.csv"),
            "runtime_notes_path": str(directories["models"] / "station_level_runtime_notes.txt"),
            "configured_model_status_path": str(directories["models"] / "station_level_configured_model_status.csv"),
            "native_tree_backends": {
                row["model_name"]: bool(row["native_backend_available"])
                for row in (
                    [] if runtime_metadata is None else runtime_metadata["configured_models"].to_dict(orient="records")
                )
                if row["model_name"] in {"lgbm", "xgboost"}
            },
        },
        directories["models"] / "station_level_run_manifest.json",
    )


def run_station_level_pipeline(
    config: StationLevelForecastConfig,
    model: str = "all",
    tune: bool | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    progress = progress or noop_progress
    directories = ensure_output_directories(config)
    runtime_metadata = _station_runtime_metadata(config)
    progress("Writing runtime diagnostics.")
    write_station_level_runtime_outputs(directories, runtime_metadata)
    progress("Preparing station-level artifacts.")
    artifacts = build_station_level_artifacts(config)
    progress(
        f"Artifacts ready: panel_rows={len(artifacts['panel'])} observed_rows={len(artifacts['observed_daily'])}."
    )
    progress("Writing feature artifacts.")
    write_station_level_feature_artifacts(directories, artifacts)

    progress("Starting backtest stage.")
    backtest_outputs = run_station_level_backtest_stage(
        config,
        artifacts["panel"],
        artifacts["slice_lookup"],
        artifacts["feature_frame"],
        model=model,
        tune=tune,
        progress=progress,
    )
    progress("Writing backtest outputs.")
    write_station_level_backtest_outputs(directories, backtest_outputs)

    progress("Starting production stage.")
    production_outputs = run_station_level_production_stage(
        config,
        artifacts["panel"],
        artifacts["slice_lookup"],
        artifacts["feature_frame"],
        backtest_outputs["interval_calibration"],
        backtest_outputs["enabled_model_keys"],
        backtest_outputs["tuning_enabled"],
        requested_model=model,
        progress=progress,
    )
    progress("Writing production outputs.")
    write_station_level_production_outputs(
        directories,
        production_outputs,
        config,
        requested_model=model,
        enabled_model_keys=backtest_outputs["enabled_model_keys"],
        tune=backtest_outputs["tuning_enabled"],
        runtime_metadata=runtime_metadata,
    )

    progress("Rendering comparison figure.")
    comparison_plot = plot_model_comparison(backtest_outputs["backtest_summary"], directories["figures"] / "station_level_model_comparison.png")
    progress("Station-level forecasting pipeline complete.")
    return {
        "panel_rows": len(artifacts["panel"]),
        "observed_rows": len(artifacts["observed_daily"]),
        "backtest_metric_rows": len(backtest_outputs["backtest_metrics"]),
        "forecast_rows": len(production_outputs["future_forecasts"]),
        "output_root": str(config.output_root),
        "comparison_plot": str(comparison_plot) if comparison_plot else None,
    }
