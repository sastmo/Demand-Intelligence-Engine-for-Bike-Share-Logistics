from __future__ import annotations

from typing import Any

import pandas as pd

from metro_bike_share_forecasting.system_level.diagnosis.config import DiagnosticConfig


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


def validate_time_index(df: pd.DataFrame, config: DiagnosticConfig) -> tuple[pd.DataFrame, dict[str, Any], dict[str, pd.DataFrame]]:
    working = _extract_time_target_frame(df, config)
    dropped_missing_values = int(working["observed_value"].isna().sum())
    grouped = working.groupby("timestamp", as_index=False)["observed_value"].sum(min_count=1)
    duplicate_timestamps = max(len(working) - len(grouped), 0)

    inferred_frequency = pd.infer_freq(grouped["timestamp"]) if len(grouped) >= 3 else None
    expected_frequency = config.expected_frequency or inferred_frequency

    if expected_frequency:
        full_index = pd.date_range(grouped["timestamp"].min(), grouped["timestamp"].max(), freq=expected_frequency)
        prepared = pd.DataFrame({"timestamp": full_index}).merge(grouped, on="timestamp", how="left")
    else:
        prepared = grouped.copy()

    prepared["missing_period_flag"] = prepared["observed_value"].isna().astype(int)
    prepared["value"] = prepared["observed_value"]
    prepared["value_filled"] = prepared["value"].interpolate(method="linear", limit_direction="both")
    prepared["value_filled"] = prepared["value_filled"].ffill().bfill().fillna(0.0)

    timestamp_diffs = grouped["timestamp"].sort_values().diff().dropna()
    irregular_spacing_count = int(timestamp_diffs.nunique() - 1) if len(timestamp_diffs) > 0 else 0
    gap_table = _build_gap_table(prepared)
    gap_distribution = (
        gap_table["missing_periods"].value_counts().sort_index().rename_axis("missing_periods").reset_index(name="gap_count")
        if not gap_table.empty
        else pd.DataFrame(columns=["missing_periods", "gap_count"])
    )

    metadata = {
        "row_count": int(len(grouped)),
        "duplicate_timestamps": int(duplicate_timestamps),
        "missing_periods": int(prepared["missing_period_flag"].sum()),
        "missing_values": int(dropped_missing_values),
        "inferred_frequency": inferred_frequency,
        "expected_frequency": expected_frequency,
        "timestamp_start": prepared["timestamp"].min(),
        "timestamp_end": prepared["timestamp"].max(),
        "time_index_reliable": bool(irregular_spacing_count == 0),
        "irregular_spacing_count": irregular_spacing_count,
    }
    return prepared, metadata, {"timestamp_gaps": gap_table, "gap_distribution": gap_distribution}
