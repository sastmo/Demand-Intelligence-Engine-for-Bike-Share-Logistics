"""Shared forecasting utilities used across forecasting scopes."""

from system_level.common.cli_utils import (
    discover_project_root,
    emit_summary,
)
from system_level.common.intervals import (
    INTERVAL_OUTPUT_COLUMNS,
    apply_calibrated_intervals,
    collect_backtest_residuals,
    evaluate_interval_quality,
    fit_interval_calibration,
)
from system_level.common.io import (
    OUTPUT_SUBDIRECTORIES,
    ensure_output_directories,
    write_dataframe,
    write_json,
)
from system_level.common.metrics import (
    bias,
    default_mase_season_length,
    mae,
    mase,
    rmse,
    seasonal_naive_scale,
)
from system_level.common.validation import (
    KNOWN_FUTURE_PREFIXES,
    assert_known_future_feature_coverage,
    time_ordered_validation_split,
    validate_known_future_external_frame,
)

__all__ = [
    "INTERVAL_OUTPUT_COLUMNS",
    "KNOWN_FUTURE_PREFIXES",
    "OUTPUT_SUBDIRECTORIES",
    "apply_calibrated_intervals",
    "assert_known_future_feature_coverage",
    "bias",
    "collect_backtest_residuals",
    "default_mase_season_length",
    "discover_project_root",
    "emit_summary",
    "ensure_output_directories",
    "evaluate_interval_quality",
    "fit_interval_calibration",
    "mae",
    "mase",
    "rmse",
    "seasonal_naive_scale",
    "time_ordered_validation_split",
    "validate_known_future_external_frame",
    "write_dataframe",
    "write_json",
]
