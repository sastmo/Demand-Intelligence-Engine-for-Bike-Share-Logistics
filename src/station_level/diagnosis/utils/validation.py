from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from station_level.diagnosis.config import StationDiagnosisConfig


class StationDiagnosisValidationError(ValueError):
    """Raised when station-level diagnosis input validation fails."""


def validate_required_columns(frame: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Raise a clear error when expected columns are missing."""

    missing = sorted(set(required_columns).difference(frame.columns))
    if missing:
        raise StationDiagnosisValidationError(f"Input data is missing required columns: {missing}")


def _summary_row(check: str, severity: str, status: str, count: int | float, message: str) -> dict[str, object]:
    return {
        "check": check,
        "severity": severity,
        "status": status,
        "count": int(count) if pd.notna(count) else np.nan,
        "message": message,
    }


def build_validation_artifacts(
    station_daily: pd.DataFrame,
    config: StationDiagnosisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate normalized station-daily data and return summary and station-level warning tables.

    Expected input columns are: date, station_id, target.
    Hard failures raise ``StationDiagnosisValidationError`` after the summary table is assembled.
    """

    issues: list[dict[str, object]] = []
    errors: list[str] = []
    warnings: list[dict[str, object]] = []

    frame = station_daily.copy()
    frame = frame.sort_values(["station_id", "date"]).reset_index(drop=True)

    invalid_date_rows = int(frame["date"].isna().sum())
    issues.append(
        _summary_row(
            "invalid_dates",
            "error",
            "fail" if invalid_date_rows else "pass",
            invalid_date_rows,
            "Rows with unparsable or missing dates are not allowed.",
        )
    )
    if invalid_date_rows:
        errors.append(f"Found {invalid_date_rows} rows with invalid dates.")

    blank_station_rows = int(frame["station_id"].astype(str).str.strip().eq("").sum())
    issues.append(
        _summary_row(
            "blank_station_ids",
            "error",
            "fail" if blank_station_rows else "pass",
            blank_station_rows,
            "Station identifiers must be non-empty after normalization.",
        )
    )
    if blank_station_rows:
        errors.append(f"Found {blank_station_rows} blank station identifiers.")

    duplicate_rows = int(frame.duplicated(subset=["station_id", "date"]).sum())
    issues.append(
        _summary_row(
            "duplicate_station_date_rows",
            "error",
            "fail" if duplicate_rows else "pass",
            duplicate_rows,
            "Each station-date pair must be unique.",
        )
    )
    if duplicate_rows:
        errors.append(f"Found {duplicate_rows} duplicate station-date rows.")

    invalid_target_rows = int(frame["target"].isna().sum())
    issues.append(
        _summary_row(
            "invalid_target_rows",
            "error",
            "fail" if invalid_target_rows else "pass",
            invalid_target_rows,
            "Target values must be numeric or null only when explicitly treated as missing observations.",
        )
    )
    if invalid_target_rows:
        errors.append(f"Found {invalid_target_rows} rows with invalid or missing target values.")

    negative_targets = int(frame["target"].lt(0).fillna(False).sum())
    issues.append(
        _summary_row(
            "negative_demand_values",
            "error",
            "fail" if negative_targets else "pass",
            negative_targets,
            "Negative demand values are invalid for daily station totals.",
        )
    )
    if negative_targets:
        errors.append(f"Found {negative_targets} rows with negative demand values.")

    all_null_station_count = int(frame.groupby("station_id")["target"].apply(lambda s: s.isna().all()).sum())
    issues.append(
        _summary_row(
            "all_null_target_stations",
            "error",
            "fail" if all_null_station_count else "pass",
            all_null_station_count,
            "Stations with no usable target values cannot be diagnosed.",
        )
    )
    if all_null_station_count:
        errors.append(f"Found {all_null_station_count} stations with all-null targets.")

    if not frame.empty and frame["target"].notna().any():
        positive_targets = frame.loc[frame["target"].notna() & (frame["target"] >= 0), "target"]
        spike_reference = max(
            float(positive_targets.quantile(config.validation_spike_quantile)),
            float(positive_targets.median() * config.validation_spike_multiplier),
        )
        suspicious_spikes = int((frame["target"] > spike_reference).sum())
    else:
        spike_reference = np.nan
        suspicious_spikes = 0
    issues.append(
        _summary_row(
            "suspicious_target_spikes",
            "warning",
            "warn" if suspicious_spikes else "pass",
            suspicious_spikes,
            f"Values above {spike_reference:.3f} are flagged as suspicious spikes for review." if pd.notna(spike_reference) else "No valid targets available for spike detection.",
        )
    )

    global_start = frame["date"].min() if frame["date"].notna().any() else pd.NaT
    global_end = frame["date"].max() if frame["date"].notna().any() else pd.NaT

    station_warning_rows: list[dict[str, object]] = []
    if pd.notna(global_start) and pd.notna(global_end):
        full_horizon_days = int((global_end - global_start).days) + 1
        for station_id, station_frame in frame.groupby("station_id", sort=True):
            observed = station_frame.sort_values("date")
            usable = observed["target"].notna().sum()
            valid_dates = observed["date"].dropna()
            missing_dates = valid_dates.diff().dt.days.sub(1).fillna(0)
            longest_gap = int(max(0, missing_dates.max())) if not missing_dates.empty else 0
            span_days = int((valid_dates.max() - valid_dates.min()).days) + 1 if not valid_dates.empty else 0
            station_warning_rows.append(
                {
                    "station_id": str(station_id),
                    "observed_days": int(usable),
                    "history_span_days": int(span_days),
                    "calendar_coverage_ratio": float(usable / span_days) if span_days > 0 else np.nan,
                    "global_presence_ratio": float(usable / full_horizon_days) if full_horizon_days > 0 else np.nan,
                    "longest_missing_gap_days": longest_gap,
                    "warning_too_few_usable_days": bool(usable < config.warning_min_usable_days),
                    "warning_long_missing_gap": bool(longest_gap >= config.warning_long_missing_streak_days),
                }
            )
        warnings = station_warning_rows

    warning_frame = pd.DataFrame(warnings)
    too_few_usable_days = int(warning_frame.get("warning_too_few_usable_days", pd.Series(dtype=bool)).sum()) if not warning_frame.empty else 0
    long_missing_gaps = int(warning_frame.get("warning_long_missing_gap", pd.Series(dtype=bool)).sum()) if not warning_frame.empty else 0
    issues.append(
        _summary_row(
            "stations_with_too_little_usable_data",
            "warning",
            "warn" if too_few_usable_days else "pass",
            too_few_usable_days,
            f"Stations with fewer than {config.warning_min_usable_days} usable observed days are flagged.",
        )
    )
    issues.append(
        _summary_row(
            "stations_with_long_missing_spans",
            "warning",
            "warn" if long_missing_gaps else "pass",
            long_missing_gaps,
            f"Stations with missing gaps of at least {config.warning_long_missing_streak_days} days are flagged.",
        )
    )

    summary = pd.DataFrame(issues)
    if errors:
        joined = " ".join(errors)
        raise StationDiagnosisValidationError(joined)
    return summary, warning_frame
