from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def serialize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): serialize_value(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_value(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def prepare_output_dirs(output_root: Path, clean_output: bool) -> tuple[Path, Path, Path]:
    if clean_output and output_root.exists():
        shutil.rmtree(output_root)
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    report_dir = output_root / "reports"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, tables_dir, report_dir


def write_tables(tables: dict[str, pd.DataFrame], tables_dir: Path) -> dict[str, Path]:
    written: dict[str, Path] = {}
    for name, frame in tables.items():
        path = tables_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        written[name] = path
    return written


def write_summary_files(summary: dict[str, Any], tables_dir: Path) -> dict[str, Path]:
    json_path = tables_dir / "diagnostics_summary.json"
    csv_path = tables_dir / "diagnostics_summary.csv"
    serialized = serialize_value(summary)
    json_path.write_text(json.dumps(serialized, indent=2))
    flat_summary: dict[str, Any] = {}
    for key, value in summary.items():
        flat_summary[key] = json.dumps(serialize_value(value)) if isinstance(value, (dict, list)) else serialize_value(value)
    pd.DataFrame([flat_summary]).to_csv(csv_path, index=False)
    return {"diagnostics_summary_json": json_path, "diagnostics_summary_csv": csv_path}


def write_markdown_report(summary: dict[str, Any], report_dir: Path) -> Path:
    lines = [
        f"# Forecasting Diagnostics Report: {summary['series_name']}",
        "",
        "## Time index reliability",
        f"- Reliable time index: {summary.get('time_index_reliable')}",
        f"- Frequency: {summary.get('frequency')}",
        f"- Missing timestamps: {summary.get('missing_periods')}",
        f"- Duplicate timestamps: {summary.get('duplicate_timestamps')}",
        f"- Irregular spacing count: {summary.get('irregular_spacing_count')}",
        "",
        "## Core findings",
    ]
    for insight in summary.get("insights", [])[:8]:
        lines.append(f"- {insight}")

    lines.extend(
        [
            "",
            "## Forecasting guidance",
        ]
    )
    for recommendation in summary.get("recommendations", []):
        lines.append(f"- {recommendation}")

    lines.extend(
        [
            "",
            "## Key risks",
        ]
    )
    for risk in summary.get("risks", []):
        lines.append(f"- {risk}")
    if not summary.get("risks"):
        lines.append("- No severe structural risk was flagged beyond the normal need for time-aware validation.")

    lines.extend(
        [
            "",
            "## Recommended model families",
        ]
    )
    for method in summary.get("recommended_model_families", []):
        lines.append(f"- {method}")

    lines.extend(
        [
            "",
            "## Key statistics",
            f"- Trend strength: {summary.get('trend_strength')}",
            f"- Seasonal strength: {summary.get('seasonal_strength')}",
            f"- Dominant periods: {summary.get('dominant_periods')}",
            f"- Lag-1 autocorrelation: {summary.get('lag1_autocorrelation')}",
            f"- Seasonal lag autocorrelation: {summary.get('seasonal_lag_autocorrelation')}",
            f"- Stationarity assessment: {summary.get('stationarity_assessment')}",
            f"- Level shifts detected: {summary.get('level_shift_count')}",
            f"- Count-like target: {summary.get('is_count_like')}",
            f"- Intermittent-like zero mass: {summary.get('is_intermittent_like')}",
        ]
    )
    path = report_dir / "diagnostics_report.md"
    path.write_text("\n".join(lines))
    return path
