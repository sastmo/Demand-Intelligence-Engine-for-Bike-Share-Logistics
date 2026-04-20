from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from system_level.common.cli_utils import discover_project_root

REPO_ROOT = discover_project_root(__file__)


SYSTEM_DIAGNOSIS_FIGURES = [
    ("Series", "series.png"),
    ("Rolling Stats", "rolling_stats.png"),
    ("Seasonal Profile", "seasonal_profile.png"),
    ("STL Decomposition", "stl.png"),
    ("Autocorrelation", "acf.png"),
    ("Partial Autocorrelation", "pacf.png"),
    ("Frequency Spectrum", "periodogram.png"),
    ("Distribution", "distribution.png"),
    ("Outliers", "outliers.png"),
    ("Time Gaps", "time_index_gaps.png"),
]

STATION_DIAGNOSIS_FIGURES = [
    ("Category Counts", "category_counts.png"),
    ("Cluster Counts", "cluster_counts.png"),
    ("History Days", "history_days_histogram.png"),
    ("Zero Rate", "zero_rate_histogram.png"),
    ("Average Demand", "avg_demand_histogram.png"),
    ("Coefficient of Variation", "coefficient_of_variation_histogram.png"),
    ("Demand vs Zero Rate", "avg_demand_vs_zero_rate_by_category.png"),
    ("Demand vs CV", "avg_demand_vs_cv_by_cluster.png"),
    ("Lag-7 vs Weekday Effect", "lag7_vs_weekday_effect_by_category.png"),
    ("Demand by Category", "avg_demand_by_category_boxplot.png"),
    ("Zero Rate by Category", "zero_rate_by_category_boxplot.png"),
    ("Cluster Profile Heatmap", "cluster_profile_heatmap.png"),
    ("History Group Counts", "history_group_counts.png"),
    ("Representative Time Series", "representative_station_timeseries.png"),
    ("Representative Weekday Profiles", "representative_weekday_profiles.png"),
    ("Representative Monthly Profiles", "representative_monthly_profiles.png"),
]


def repo_root() -> Path:
    return REPO_ROOT


def read_markdown(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def read_image_manifest(base_dir: Path, items: list[tuple[str, str]]) -> list[tuple[str, Path]]:
    manifest: list[tuple[str, Path]] = []
    for label, filename in items:
        path = base_dir / filename
        if path.exists():
            manifest.append((label, path))
    return manifest


def diagnosis_bundle(level: str) -> dict[str, object]:
    base = repo_root() / "diagnosis" / f"{level}_level"
    tables_dir = base / "outputs" / "tables"
    figures_dir = base / "outputs" / "figures"
    note_path = base / "analysis_insights.md"
    figure_manifest = SYSTEM_DIAGNOSIS_FIGURES if level == "system" else STATION_DIAGNOSIS_FIGURES

    if level == "system":
        tables = {
            "summary": read_table(tables_dir / "diagnostics_summary.csv"),
            "baseline": read_table(tables_dir / "baseline_summary.csv"),
            "weekday_profile": read_table(tables_dir / "weekday_profile.csv"),
            "monthly_profile": read_table(tables_dir / "monthly_profile.csv"),
            "outliers": read_table(tables_dir / "outlier_candidates.csv"),
            "level_shifts": read_table(tables_dir / "level_shifts.csv"),
            "gaps": read_table(tables_dir / "gap_distribution.csv"),
        }
    else:
        tables = {
            "inventory": read_table(tables_dir / "station_inventory.csv"),
            "summary": read_table(tables_dir / "station_summary_table.csv"),
            "summary_with_clusters": read_table(tables_dir / "station_summary_with_clusters.csv"),
            "category_summary": read_table(tables_dir / "station_category_summary.csv"),
            "cluster_profile": read_table(tables_dir / "station_cluster_profile.csv"),
            "cluster_selection": read_table(tables_dir / "cluster_model_selection.csv"),
            "top_avg_demand": read_table(tables_dir / "top_by_avg_demand.csv"),
            "top_zero_rate": read_table(tables_dir / "top_by_zero_rate.csv"),
            "top_outlier_rate": read_table(tables_dir / "top_by_outlier_rate.csv"),
            "top_cv": read_table(tables_dir / "top_by_coefficient_of_variation.csv"),
        }

    return {
        "base": base,
        "tables": tables,
        "note_path": note_path,
        "note_text": read_markdown(note_path),
        "figures": read_image_manifest(figures_dir, figure_manifest),
    }


def forecast_bundle(level: str) -> dict[str, object]:
    base = repo_root() / "forecasts" / f"{level}_level"
    feature_dir = base / "feature_artifacts"
    backtest_dir = base / "backtests"
    metrics_dir = base / "metrics"
    models_dir = base / "models"
    forecast_dir = base / "forecasts"
    figures_dir = base / "figures"
    note_path = base / "forecasting_contract.md" if level == "system" else None

    if level == "system":
        tables = {
            "comparison": read_table(metrics_dir / "system_level_model_comparison.csv"),
            "future": read_table(forecast_dir / "system_level_future_forecasts.csv"),
            "backtest_metrics": read_table(backtest_dir / "system_level_fold_metrics.csv"),
            "backtest_forecasts": read_table(backtest_dir / "system_level_fold_forecasts.csv"),
            "residuals": read_table(backtest_dir / "system_level_backtest_residuals.csv"),
            "interval_coverage": read_table(metrics_dir / "system_level_interval_coverage.csv"),
            "interval_coverage_by_step": read_table(metrics_dir / "system_level_interval_coverage_by_step.csv"),
            "interval_calibration": read_table(metrics_dir / "system_level_interval_calibration.csv"),
            "registry": read_table(models_dir / "system_level_model_registry.csv"),
            "fit_diagnostics": read_table(models_dir / "system_level_production_fit_diagnostics.csv"),
            "target": read_table(feature_dir / "system_level_target.csv"),
            "features": read_table(feature_dir / "system_level_features.csv"),
        }
        figures = read_image_manifest(
            figures_dir,
            [
                ("Model Comparison", "system_level_model_comparison.png"),
                ("Future Forecasts", "system_level_future_forecasts.png"),
            ],
        )
    else:
        tables = {
            "comparison": read_table(metrics_dir / "station_level_model_comparison.csv"),
            "future": read_table(forecast_dir / "station_level_future_forecasts.csv"),
            "slice_metrics": read_table(metrics_dir / "station_level_slice_metrics.csv"),
            "backtest_metrics": read_table(backtest_dir / "station_level_fold_metrics.csv"),
            "backtest_forecasts": read_table(backtest_dir / "station_level_fold_forecasts.csv"),
            "windows": read_table(backtest_dir / "station_level_backtest_windows.csv"),
            "residuals": read_table(backtest_dir / "station_level_backtest_residuals.csv"),
            "interval_coverage": read_table(metrics_dir / "station_level_interval_coverage.csv"),
            "interval_coverage_by_step": read_table(metrics_dir / "station_level_interval_coverage_by_step.csv"),
            "interval_calibration": read_table(metrics_dir / "station_level_interval_calibration.csv"),
            "registry": read_table(models_dir / "station_level_model_registry.csv"),
            "observed_panel": read_table(feature_dir / "station_level_observed_panel.csv"),
            "slice_lookup": read_table(feature_dir / "station_level_slice_lookup.csv"),
        }
        figures = read_image_manifest(
            figures_dir,
            [("Model Comparison", "station_level_model_comparison.png")],
        )

    return {
        "base": base,
        "tables": tables,
        "manifest": read_json(models_dir / f"{level}_level_run_manifest.json"),
        "note_path": note_path,
        "note_text": read_markdown(note_path) if note_path else None,
        "figures": figures,
    }


def format_short_number(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    value = float(value)
    magnitude = abs(value)
    if magnitude >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    if magnitude >= 1_000:
        return f"{value/1_000:.1f}K"
    if value.is_integer():
        return f"{int(value)}"
    return f"{value:.2f}"


def prediction_column(frame: pd.DataFrame) -> str:
    return "point_forecast" if "point_forecast" in frame.columns else "prediction"


def station_forecast_chart_frame(
    observed_panel: pd.DataFrame,
    future_forecasts: pd.DataFrame,
    station_id: str,
    model_name: str,
    horizon: int,
    history_days: int = 90,
) -> pd.DataFrame:
    history = observed_panel.loc[observed_panel["station_id"].astype(str) == str(station_id)].copy()
    if "date" in history.columns:
        history["date"] = pd.to_datetime(history["date"])
    history = history.sort_values("date").tail(history_days)

    forecast = future_forecasts.loc[
        (future_forecasts["station_id"].astype(str) == str(station_id))
        & (future_forecasts["model_name"].astype(str) == str(model_name))
        & (future_forecasts["horizon"] == horizon)
    ].copy()
    if "date" in forecast.columns:
        forecast["date"] = pd.to_datetime(forecast["date"])
    forecast = forecast.sort_values("date")
    value_col = prediction_column(forecast)

    history_frame = history[["date", "target"]].rename(columns={"target": "observed"})
    forecast_frame = forecast[["date", value_col, "lower_80", "upper_80"]].rename(columns={value_col: "forecast"})
    chart = history_frame.merge(forecast_frame, on="date", how="outer").sort_values("date")
    return chart


def system_forecast_chart_frame(future_forecasts: pd.DataFrame, model_name: str, horizon: int) -> pd.DataFrame:
    forecast = future_forecasts.loc[
        (future_forecasts["model_name"].astype(str) == str(model_name))
        & (future_forecasts["horizon"] == horizon)
    ].copy()
    if "date" in forecast.columns:
        forecast["date"] = pd.to_datetime(forecast["date"])
    value_col = prediction_column(forecast)
    return forecast[["date", value_col, "lower_80", "upper_80"]].rename(columns={value_col: "forecast"}).sort_values("date")
