from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from system_level.diagnosis.anomalies import detect_anomalies
from system_level.diagnosis.autocorrelation import summarize_autocorrelation
from system_level.diagnosis.baselines import compute_baseline_diagnostics
from system_level.diagnosis.config import DiagnosticConfig, apply_frequency_defaults, base_frequency_label
from system_level.diagnosis.distribution import summarize_distribution
from system_level.diagnosis.frequency import summarize_frequency_domain
from system_level.diagnosis.plotting import (
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
from system_level.diagnosis.seasonality import build_profile_tables, choose_primary_period, detect_multiple_seasonality
from system_level.diagnosis.stationarity import run_stationarity_checks
from system_level.diagnosis.time_index import validate_time_index
from system_level.diagnosis.trend import analyze_trend_and_decomposition, detect_level_shifts
from system_level.diagnosis.types import DiagnosticResult


DIAGNOSTIC_SERIES_USAGE = [
    {
        "diagnostic_name": "time_index_validation",
        "series_used": "observed_and_imputed_tracking",
        "retrospective": False,
        "screening_only": False,
        "notes": "Tracks observed values, missing periods, and filled cadence values separately.",
    },
    {
        "diagnostic_name": "baseline_screening",
        "series_used": "filled_cadence_series",
        "retrospective": True,
        "screening_only": True,
        "notes": "In-sample naive baselines are descriptive screens, not forecast validation.",
    },
    {
        "diagnostic_name": "trend_and_decomposition",
        "series_used": "filled_cadence_series",
        "retrospective": True,
        "screening_only": False,
        "notes": "Seasonal decomposition requires a contiguous cadence and may rely on imputed points.",
    },
    {
        "diagnostic_name": "frequency_domain",
        "series_used": "filled_cadence_series",
        "retrospective": True,
        "screening_only": False,
        "notes": "Spectral diagnostics are descriptive and frequency-aware, not forecast validation.",
    },
    {
        "diagnostic_name": "autocorrelation",
        "series_used": "filled_cadence_series",
        "retrospective": True,
        "screening_only": False,
        "notes": "Autocorrelation diagnostics assume a regular cadence.",
    },
    {
        "diagnostic_name": "anomaly_detection",
        "series_used": "filled_cadence_series_with_imputed_points_suppressed",
        "retrospective": True,
        "screening_only": False,
        "notes": "Centered anomaly detection is retrospective by default; imputed-point anomalies are suppressed.",
    },
    {
        "diagnostic_name": "distribution_summary",
        "series_used": "observed_series",
        "retrospective": False,
        "screening_only": False,
        "notes": "Distribution summaries are based on observed values only.",
    },
    {
        "diagnostic_name": "seasonal_profiles",
        "series_used": "observed_series_plus_filled_reference",
        "retrospective": False,
        "screening_only": False,
        "notes": "Profile tables expose both observed-only and filled-series averages.",
    },
    {
        "diagnostic_name": "stationarity_tests",
        "series_used": "filled_cadence_series",
        "retrospective": True,
        "screening_only": True,
        "notes": "ADF and KPSS are screening tests only and should not be overinterpreted.",
    },
    {
        "diagnostic_name": "level_shift_detection",
        "series_used": "filled_cadence_series",
        "retrospective": True,
        "screening_only": False,
        "notes": "Level-shift detection is retrospective and optional-dependency-backed.",
    },
]


def _prepare_output_dirs(output_root: Path, clean_output: bool) -> tuple[Path, Path, Path]:
    if clean_output and output_root.exists():
        shutil.rmtree(output_root)
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    report_dir = output_root / "report"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, tables_dir, report_dir


def _write_tables(tables: dict[str, pd.DataFrame], tables_dir: Path) -> dict[str, Path]:
    written: dict[str, Path] = {}
    for name, frame in tables.items():
        path = tables_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        written[name] = path
    return written


def _write_summary_artifacts(summary: dict[str, object], tables_dir: Path) -> dict[str, Path]:
    flat_summary: dict[str, object] = {}
    for key, value in summary.items():
        flat_summary[key] = value if not isinstance(value, (dict, list)) else str(value)
    csv_path = tables_dir / "diagnostics_summary.csv"
    json_path = tables_dir / "diagnostics_summary.json"
    pd.DataFrame([flat_summary]).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2, default=str))
    return {
        "diagnostics_summary_csv": csv_path,
        "diagnostics_summary_json": json_path,
    }


def _build_dependency_status_table(summary: dict[str, object]) -> pd.DataFrame:
    rows = [
        {
            "dependency_name": "mstl",
            "available": bool(summary.get("mstl_available", False)),
            "status": summary.get("decomposition_status"),
            "reason": summary.get("decomposition_reason"),
        },
        {
            "dependency_name": "ruptures",
            "available": bool(summary.get("level_shift_detection_available", False)),
            "status": summary.get("level_shift_detection_status"),
            "reason": summary.get("level_shift_detection_reason"),
        },
        {
            "dependency_name": "adf",
            "available": summary.get("adf_status") == "ok",
            "status": summary.get("adf_status"),
            "reason": "ADF is used as a screening test only.",
        },
        {
            "dependency_name": "kpss",
            "available": summary.get("kpss_status") == "ok",
            "status": summary.get("kpss_status"),
            "reason": "KPSS is used as a screening test only.",
        },
    ]
    return pd.DataFrame(rows)


def _build_report(summary: dict[str, object], report_dir: Path) -> Path:
    report_path = report_dir / "diagnostics_report.md"
    lines = [
        f"# {summary['series_name']} Diagnosis Report",
        "",
        "## Scope",
        "- This report is descriptive diagnosis for a single system-level series.",
        "- It is not a forecasting backtest report.",
        "",
        "## Time Index",
        f"- Frequency label: {summary.get('frequency')}",
        f"- Observed points: {summary.get('observed_points')}",
        f"- Imputed points: {summary.get('imputed_points')}",
        f"- Missing periods: {summary.get('missing_periods')}",
        f"- Imputation method: {summary.get('imputation_method')}",
        "",
        "## Screening Caveats",
        f"- Baseline scope: {summary.get('baseline_screening_scope')}",
        f"- Stationarity scope: {summary.get('stationarity_test_status')}",
        f"- Anomaly scope: {summary.get('anomaly_scope')}",
        "",
        "## Decomposition And Dependencies",
        f"- Decomposition method: {summary.get('decomposition_method')}",
        f"- Decomposition status: {summary.get('decomposition_status')}",
        f"- MSTL available: {summary.get('mstl_available')}",
        f"- Level-shift detection status: {summary.get('level_shift_detection_status')}",
        "",
        "## Series Usage",
        f"- Diagnostics using imputed series: {', '.join(summary.get('diagnostics_using_imputed_series', [])) or 'none'}",
        f"- Diagnostics using observed-only series: {', '.join(summary.get('diagnostics_using_observed_only', [])) or 'none'}",
    ]
    warnings = summary.get("warnings", [])
    if warnings:
        lines.extend(["", "## Warnings"])
        lines.extend([f"- {warning}" for warning in warnings])
    report_path.write_text("\n".join(lines))
    return report_path


def _build_warning_list(summary: dict[str, object]) -> list[str]:
    warnings: list[str] = []
    if int(summary.get("imputed_points", 0) or 0) > 0:
        warnings.append(
            "Some diagnostics relied on a filled cadence series because missing periods or missing observations were present."
        )
    if summary.get("level_shift_detection_status") == "unavailable_optional_dependency":
        warnings.append("Level-shift detection was skipped because `ruptures` is not installed.")
    if summary.get("decomposition_status") not in {"ok"}:
        warnings.append(str(summary.get("decomposition_reason")))
    if summary.get("stationarity_test_status") != "ok":
        warnings.append("Stationarity screening was partial or unavailable; interpret it cautiously.")
    return warnings


def run_forecasting_diagnostics(df: pd.DataFrame, config: DiagnosticConfig) -> DiagnosticResult:
    config = apply_frequency_defaults(config)
    figures_dir, tables_dir, report_dir = _prepare_output_dirs(Path(config.output_root), config.clean_output)

    # Time-index validation comes first because every downstream forecast diagnostic
    # depends on a trustworthy cadence and an explicit view of missing periods.
    prepared, time_index_summary, time_index_tables = validate_time_index(df, config)
    primary_period = choose_primary_period(config.candidate_periods, len(prepared), config.primary_period)
    rolling_window = config.rolling_window or max(primary_period or 7, 7)
    max_acf_lags = config.max_acf_lags or max(24, (primary_period or 12) * 2)

    observed_values = prepared["observed_value"]
    filled_values = prepared["value_filled"]
    # Baselines anchor the problem. If seasonal naive is already strong, complex models
    # need to explain why they should beat it.
    baseline_summary, baseline_table = compute_baseline_diagnostics(filled_values, primary_period)
    # Trend/decomposition and level shifts tell us whether stable-level methods are
    # plausible or whether regime-aware logic is required.
    trend_summary, decomposition = analyze_trend_and_decomposition(filled_values, config.candidate_periods, primary_period)
    trend_component = pd.Series(np.asarray(decomposition.trend), index=prepared.index) if decomposition is not None and hasattr(decomposition, "trend") else None
    frequency_summary = summarize_frequency_domain(filled_values, config.candidate_periods)
    multiple_seasonalities = detect_multiple_seasonality(frequency_summary, config.candidate_periods)
    # ACF/PACF and outlier behavior are key for deciding between classical residual models,
    # lag-feature models, and more robust probabilistic approaches.
    autocorrelation_summary = summarize_autocorrelation(filled_values, primary_period)
    outlier_detail, outlier_summary = detect_anomalies(
        filled_values,
        rolling_window,
        config.outlier_threshold,
        method=config.anomaly_method,
        is_imputed=prepared["imputed_flag"],
    )
    distribution_summary = summarize_distribution(observed_values)
    stationarity_summary = run_stationarity_checks(filled_values)
    level_shifts, level_shift_summary = detect_level_shifts(filled_values, prepared["timestamp"])
    profiles = build_profile_tables(prepared, base_frequency_label(config.frequency))
    diagnostic_usage_table = pd.DataFrame(DIAGNOSTIC_SERIES_USAGE)
    diagnostics_using_imputed_series = diagnostic_usage_table.loc[
        diagnostic_usage_table["series_used"].str.contains("filled", na=False), "diagnostic_name"
    ].tolist()
    diagnostics_using_observed_only = diagnostic_usage_table.loc[
        diagnostic_usage_table["series_used"].isin(["observed_series"]),
        "diagnostic_name",
    ].tolist()
    diagnostics_using_observed_and_filled = diagnostic_usage_table.loc[
        diagnostic_usage_table["series_used"].isin(["observed_series_plus_filled_reference"]),
        "diagnostic_name",
    ].tolist()

    summary: dict[str, object] = {
        "summary_version": "diagnosis_v2",
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
        **level_shift_summary,
        "primary_period": primary_period,
        "candidate_periods": list(config.candidate_periods),
        "multiple_seasonalities_detected": multiple_seasonalities,
        "diagnostics_using_imputed_series": diagnostics_using_imputed_series,
        "diagnostics_using_observed_only": diagnostics_using_observed_only,
        "diagnostics_using_observed_and_filled": diagnostics_using_observed_and_filled,
        "level_shift_count": len(level_shifts),
        "level_shifts": [{"label": item["label"], "timestamp": pd.Timestamp(item["timestamp"]).isoformat()} for item in level_shifts],
        "event_markers": [{"label": event.label, "timestamp": event.timestamp.isoformat(), "color": event.color} for event in config.events],
    }
    summary["warnings"] = _build_warning_list(summary)

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
    save_decomposition_plot(decomposition, prepared, figure_paths["stl"], trend_summary)
    save_profile_plot(profiles, figure_paths["seasonal_profile"])
    save_acf_pacf(filled_values, max_acf_lags, figure_paths["acf"], figure_paths["pacf"])
    save_periodogram_plot(filled_values, figure_paths["periodogram"])
    save_distribution_plot(observed_values, figure_paths["distribution"])
    save_outlier_plot(prepared, outlier_detail, figure_paths["outliers"])

    tables = {
        **time_index_tables,
        **profiles,
        "baseline_summary": baseline_table,
        "outlier_candidates": pd.concat(
            [
                prepared[["timestamp", "observed_value", "value_filled", "imputed_flag", "value_source"]],
                outlier_detail,
            ],
            axis=1,
        ),
        "level_shifts": pd.DataFrame(summary["level_shifts"], columns=["label", "timestamp"]),
        "diagnostic_series_usage": diagnostic_usage_table,
        "dependency_status": _build_dependency_status_table(summary),
    }
    table_paths = _write_tables(tables, tables_dir)
    table_paths.update(_write_summary_artifacts(summary, tables_dir))
    report_path = _build_report(summary, report_dir)

    return DiagnosticResult(
        summary=summary,
        output_root=Path(config.output_root),
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        report_dir=report_dir,
        figures=figure_paths,
        tables=table_paths,
        report_path=report_path,
        warnings=list(summary["warnings"]),
    )
