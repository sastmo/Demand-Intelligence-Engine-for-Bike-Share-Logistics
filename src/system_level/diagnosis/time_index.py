from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from system_level.diagnosis.config import DiagnosticConfig, FREQUENCY_DEFAULTS, base_frequency_label


def _extract_time_target_frame(df: pd.DataFrame, config: DiagnosticConfig) -> pd.DataFrame:
    if config.time_col:
        working = df[[config.time_col, config.target_col]].copy()
        working = working.rename(columns={config.time_col: "timestamp", config.target_col: "observed_value"})
    elif isinstance(df.index, pd.DatetimeIndex):
        working = df[[config.target_col]].copy().reset_index()
        working = working.rename(columns={working.columns[0]: "timestamp", config.target_col: "observed_value"})
    else:
        raise ValueError("Provide `time_col` or a DatetimeIndex so the diagnostics can validate the time axis.")

    working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce")
    working["observed_value"] = pd.to_numeric(working["observed_value"], errors="coerce")
    working = working.dropna(subset=["timestamp"]).sort_values("timestamp")
    return working


def _build_gap_table(prepared: pd.DataFrame) -> pd.DataFrame:
    missing = prepared.loc[prepared["missing_period_flag"] == 1, ["timestamp"]].copy()
    if missing.empty:
        return pd.DataFrame(columns=["gap_start", "gap_end", "missing_periods"])

    missing = missing.reset_index().rename(columns={"index": "original_index"})
    gap_ids = missing["original_index"].diff().fillna(1).ne(1).cumsum()
    grouped = (
        missing.assign(gap_id=gap_ids)
        .groupby("gap_id", as_index=False)
        .agg(
            gap_start=("timestamp", "min"),
            gap_end=("timestamp", "max"),
            missing_periods=("timestamp", "count"),
        )
    )
    return grouped[["gap_start", "gap_end", "missing_periods"]]


def _fill_series(values: pd.Series, method: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    if method == "linear_interpolate_then_edge_fill":
        filled = numeric.interpolate(method="linear", limit_direction="both")
        return filled.ffill().bfill().fillna(0.0)
    raise ValueError(f"Unsupported imputation method: {method}")


def validate_time_index(df: pd.DataFrame, config: DiagnosticConfig) -> tuple[pd.DataFrame, dict[str, Any], dict[str, pd.DataFrame]]:
    working = _extract_time_target_frame(df, config)
    dropped_missing_values = int(working["observed_value"].isna().sum())
    grouped = working.groupby("timestamp", as_index=False)["observed_value"].sum(min_count=1)
    duplicate_timestamps = max(len(working) - len(grouped), 0)

    inferred_frequency = pd.infer_freq(grouped["timestamp"]) if len(grouped) >= 3 else None
    fallback_expected_frequency = FREQUENCY_DEFAULTS.get(base_frequency_label(config.frequency), {}).get("expected_frequency")
    expected_frequency = config.expected_frequency or fallback_expected_frequency or inferred_frequency

    if expected_frequency:
        full_index = pd.date_range(grouped["timestamp"].min(), grouped["timestamp"].max(), freq=expected_frequency)
        prepared = pd.DataFrame({"timestamp": full_index}).merge(grouped, on="timestamp", how="left")
    else:
        prepared = grouped.copy()

    original_timestamps = set(grouped["timestamp"])
    prepared["missing_period_flag"] = (~prepared["timestamp"].isin(original_timestamps)).astype(int)
    prepared["missing_observation_flag"] = (
        prepared["timestamp"].isin(original_timestamps) & prepared["observed_value"].isna()
    ).astype(int)
    prepared["observed_value_flag"] = prepared["observed_value"].notna().astype(int)
    prepared["imputed_flag"] = prepared["observed_value"].isna().astype(int)
    prepared["value_filled"] = _fill_series(prepared["observed_value"], config.imputation_method)
    prepared["imputed_value"] = prepared["value_filled"].where(prepared["imputed_flag"] == 1)
    prepared["value_source"] = np.where(prepared["imputed_flag"] == 1, "imputed", "observed")

    timestamp_diffs = grouped["timestamp"].sort_values().diff().dropna()
    irregular_spacing_count = int(timestamp_diffs.nunique() - 1) if len(timestamp_diffs) > 0 else 0
    gap_table = _build_gap_table(prepared)
    gap_distribution = (
        gap_table["missing_periods"].value_counts().sort_index().rename_axis("missing_periods").reset_index(name="gap_count")
        if not gap_table.empty
        else pd.DataFrame(columns=["missing_periods", "gap_count"])
    )
    prepared_table = prepared[
        [
            "timestamp",
            "observed_value",
            "value_filled",
            "imputed_value",
            "observed_value_flag",
            "missing_period_flag",
            "missing_observation_flag",
            "imputed_flag",
            "value_source",
        ]
    ].copy()

    metadata = {
        "row_count": int(len(grouped)),
        "original_row_count": int(len(working)),
        "duplicate_timestamps": int(duplicate_timestamps),
        "missing_periods": int(prepared["missing_period_flag"].sum()),
        "missing_values": int(dropped_missing_values),
        "missing_observations": int(prepared["missing_observation_flag"].sum()),
        "observed_points": int(prepared["observed_value_flag"].sum()),
        "imputed_points": int(prepared["imputed_flag"].sum()),
        "imputed_share": float(prepared["imputed_flag"].mean()) if len(prepared) else 0.0,
        "imputation_method": config.imputation_method,
        "inferred_frequency": inferred_frequency,
        "expected_frequency": expected_frequency,
        "timestamp_start": prepared["timestamp"].min(),
        "timestamp_end": prepared["timestamp"].max(),
        "time_index_reliable": bool(irregular_spacing_count == 0),
        "irregular_spacing_count": irregular_spacing_count,
    }
    return prepared, metadata, {
        "prepared_time_index": prepared_table,
        "timestamp_gaps": gap_table,
        "gap_distribution": gap_distribution,
    }
