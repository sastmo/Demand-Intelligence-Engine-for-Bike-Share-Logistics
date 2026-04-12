from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from forecasting_diagnostics.anomalies import detect_anomalies
from forecasting_diagnostics.autocorrelation import summarize_autocorrelation
from forecasting_diagnostics.baselines import compute_baseline_diagnostics
from forecasting_diagnostics.config import DiagnosticConfig, apply_frequency_defaults, base_frequency_label
from forecasting_diagnostics.distribution import summarize_distribution
from forecasting_diagnostics.frequency import summarize_frequency_domain
from forecasting_diagnostics.guidance import build_insights, build_model_guidance
from forecasting_diagnostics.plotting import (
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
from forecasting_diagnostics.report import prepare_output_dirs, write_markdown_report, write_summary_files, write_tables
from forecasting_diagnostics.seasonality import build_profile_tables, choose_primary_period, detect_multiple_seasonality
from forecasting_diagnostics.stationarity import run_stationarity_checks
from forecasting_diagnostics.time_index import validate_time_index
from forecasting_diagnostics.trend import analyze_trend_and_decomposition, detect_level_shifts
from forecasting_diagnostics.types import DiagnosticResult


def run_forecasting_diagnostics(df: pd.DataFrame, config: DiagnosticConfig) -> DiagnosticResult:
    config = apply_frequency_defaults(config)
    figures_dir, tables_dir, report_dir = prepare_output_dirs(Path(config.output_root), config.clean_output)

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
    summary["insights"] = build_insights(summary)
    recommended_models, recommendations, risks, recommendation_scores = build_model_guidance(summary)
    summary["recommended_model_families"] = recommended_models
    summary["recommendations"] = recommendations
    summary["risks"] = risks
    summary["recommendation_scores"] = recommendation_scores

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
    table_paths = write_tables(tables, tables_dir)
    table_paths.update(write_summary_files(summary, tables_dir))
    report_path = write_markdown_report(summary, report_dir)

    return DiagnosticResult(
        summary=summary,
        output_root=Path(config.output_root),
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        report_dir=report_dir,
        figures=figure_paths,
        tables=table_paths,
        report_path=report_path,
    )
