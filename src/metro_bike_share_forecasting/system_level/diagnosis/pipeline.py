from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from metro_bike_share_forecasting.system_level.diagnosis.anomalies import detect_anomalies
from metro_bike_share_forecasting.system_level.diagnosis.autocorrelation import summarize_autocorrelation
from metro_bike_share_forecasting.system_level.diagnosis.baselines import compute_baseline_diagnostics
from metro_bike_share_forecasting.system_level.diagnosis.config import DiagnosticConfig, apply_frequency_defaults, base_frequency_label
from metro_bike_share_forecasting.system_level.diagnosis.distribution import summarize_distribution
from metro_bike_share_forecasting.system_level.diagnosis.frequency import summarize_frequency_domain
from metro_bike_share_forecasting.system_level.diagnosis.plotting import (
    save_acf_pacf,
    save_decomposition_plot,
    save_distribution_plot,
    save_gap_plot,
    save_outlier_plot,
    save_periodogram_plot,
    save_profile_plot,
    save_rolling_stats_plot,
    save_series_plot,
)
from metro_bike_share_forecasting.system_level.diagnosis.seasonality import build_profile_tables, choose_primary_period, detect_multiple_seasonality
from metro_bike_share_forecasting.system_level.diagnosis.stationarity import run_stationarity_checks
from metro_bike_share_forecasting.system_level.diagnosis.time_index import validate_time_index
from metro_bike_share_forecasting.system_level.diagnosis.trend import analyze_trend_and_decomposition, detect_level_shifts
from metro_bike_share_forecasting.system_level.diagnosis.types import DiagnosticResult


def _prepare_output_dirs(output_root: Path, clean_output: bool) -> tuple[Path, Path]:
    if clean_output and output_root.exists():
        shutil.rmtree(output_root)
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, tables_dir


def _write_tables(tables: dict[str, pd.DataFrame], tables_dir: Path) -> dict[str, Path]:
    written: dict[str, Path] = {}
    for name, frame in tables.items():
        path = tables_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        written[name] = path
    return written


def _write_summary_csv(summary: dict[str, object], tables_dir: Path) -> dict[str, Path]:
    flat_summary: dict[str, object] = {}
    for key, value in summary.items():
        flat_summary[key] = value if not isinstance(value, (dict, list)) else str(value)
    path = tables_dir / "diagnostics_summary.csv"
    pd.DataFrame([flat_summary]).to_csv(path, index=False)
    return {"diagnostics_summary_csv": path}


def run_forecasting_diagnostics(df: pd.DataFrame, config: DiagnosticConfig) -> DiagnosticResult:
    config = apply_frequency_defaults(config)
    figures_dir, tables_dir = _prepare_output_dirs(Path(config.output_root), config.clean_output)

    # Time-index validation comes first because every downstream forecast diagnostic
    # depends on a trustworthy cadence and an explicit view of missing periods.
    prepared, time_index_summary, time_index_tables = validate_time_index(df, config)
    primary_period = choose_primary_period(config.candidate_periods, len(prepared), config.primary_period)
    rolling_window = config.rolling_window or max(primary_period or 7, 7)
    max_acf_lags = config.max_acf_lags or max(24, (primary_period or 12) * 2)

    values = prepared["value_filled"]
    # Baselines anchor the problem. If seasonal naive is already strong, complex models
    # need to explain why they should beat it.
    baseline_summary, baseline_table = compute_baseline_diagnostics(values, primary_period)
    # Trend/decomposition and level shifts tell us whether stable-level methods are
    # plausible or whether regime-aware logic is required.
    trend_summary, decomposition = analyze_trend_and_decomposition(values, config.candidate_periods, primary_period)
    trend_component = pd.Series(np.asarray(decomposition.trend), index=prepared.index) if decomposition is not None and hasattr(decomposition, "trend") else None
    frequency_summary = summarize_frequency_domain(values)
    multiple_seasonalities = detect_multiple_seasonality(frequency_summary["dominant_periods"], config.candidate_periods)
    # ACF/PACF and outlier behavior are key for deciding between classical residual models,
    # lag-feature models, and more robust probabilistic approaches.
    autocorrelation_summary = summarize_autocorrelation(values, primary_period)
    outlier_detail, outlier_summary = detect_anomalies(values, rolling_window, config.outlier_threshold)
    distribution_summary = summarize_distribution(values)
    stationarity_summary = run_stationarity_checks(values)
    level_shifts = detect_level_shifts(values, prepared["timestamp"])
    profiles = build_profile_tables(prepared, base_frequency_label(config.frequency))

    summary: dict[str, object] = {
        "series_name": config.series_name,
        "frequency": base_frequency_label(config.frequency) or time_index_summary.get("inferred_frequency"),
        "series_key": config.series_name,
        **time_index_summary,
        **baseline_summary,
        **trend_summary,
        **autocorrelation_summary,
        **frequency_summary,
        **outlier_summary,
        **distribution_summary,
        **stationarity_summary,
        "primary_period": primary_period,
        "candidate_periods": list(config.candidate_periods),
        "multiple_seasonalities_detected": multiple_seasonalities,
        "level_shift_count": len(level_shifts),
        "level_shifts": [{"label": item["label"], "timestamp": pd.Timestamp(item["timestamp"]).isoformat()} for item in level_shifts],
        "event_markers": [{"label": event.label, "timestamp": event.timestamp.isoformat(), "color": event.color} for event in config.events],
    }

    figure_paths = {
        "series": figures_dir / "series.png",
        "time_index_gaps": figures_dir / "time_index_gaps.png",
        "rolling_stats": figures_dir / "rolling_stats.png",
        "stl": figures_dir / "stl.png",
        "seasonal_profile": figures_dir / "seasonal_profile.png",
        "acf": figures_dir / "acf.png",
        "pacf": figures_dir / "pacf.png",
        "periodogram": figures_dir / "periodogram.png",
        "distribution": figures_dir / "distribution.png",
        "outliers": figures_dir / "outliers.png",
    }
    save_series_plot(prepared, trend_component, config, figure_paths["series"])
    save_gap_plot(prepared, figure_paths["time_index_gaps"])
    save_rolling_stats_plot(prepared, rolling_window, figure_paths["rolling_stats"])
    save_decomposition_plot(decomposition, prepared, figure_paths["stl"])
    save_profile_plot(profiles, figure_paths["seasonal_profile"])
    save_acf_pacf(values, max_acf_lags, figure_paths["acf"], figure_paths["pacf"])
    save_periodogram_plot(values, figure_paths["periodogram"])
    save_distribution_plot(values, figure_paths["distribution"])
    save_outlier_plot(prepared, outlier_detail, figure_paths["outliers"])

    tables = {
        **time_index_tables,
        **profiles,
        "baseline_summary": baseline_table,
        "outlier_candidates": pd.concat([prepared[["timestamp", "value_filled"]], outlier_detail], axis=1),
        "level_shifts": pd.DataFrame(summary["level_shifts"]),
    }
    table_paths = _write_tables(tables, tables_dir)
    table_paths.update(_write_summary_csv(summary, tables_dir))

    return DiagnosticResult(
        summary=summary,
        output_root=Path(config.output_root),
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        report_dir=None,
        figures=figure_paths,
        tables=table_paths,
        report_path=None,
    )
