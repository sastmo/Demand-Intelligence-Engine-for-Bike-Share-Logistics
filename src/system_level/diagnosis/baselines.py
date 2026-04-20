from __future__ import annotations

from typing import Any

import pandas as pd


def compute_baseline_diagnostics(values: pd.Series, primary_period: int | None) -> tuple[dict[str, Any], pd.DataFrame]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    naive_mae = None
    seasonal_naive_mae = None

    rows: list[dict[str, Any]] = []
    if len(numeric) >= 2:
        naive_errors = (numeric.iloc[1:].to_numpy() - numeric.iloc[:-1].to_numpy()).astype(float)
        naive_mae = float(abs(naive_errors).mean())
        rows.append(
            {
                "diagnostic_name": "naive_screen",
                "baseline_name": "naive",
                "diagnostic_scope": "screening_only_in_sample",
                "series_used": "filled_cadence_series",
                "mae": naive_mae,
                "season_length": None,
            }
        )

    if primary_period and len(numeric) > primary_period:
        seasonal_errors = (numeric.iloc[primary_period:].to_numpy() - numeric.iloc[:-primary_period].to_numpy()).astype(float)
        seasonal_naive_mae = float(abs(seasonal_errors).mean())
        rows.append(
            {
                "diagnostic_name": "seasonal_naive_screen",
                "baseline_name": "seasonal_naive",
                "diagnostic_scope": "screening_only_in_sample",
                "series_used": "filled_cadence_series",
                "mae": seasonal_naive_mae,
                "season_length": primary_period,
            }
        )

    if naive_mae is not None and seasonal_naive_mae is not None and naive_mae > 0:
        seasonal_gain = float((naive_mae - seasonal_naive_mae) / naive_mae)
    else:
        seasonal_gain = None

    return {
        "baseline_screening_scope": "in_sample_screening_only",
        "baseline_screening_note": (
            "These naive and seasonal-naive errors are descriptive screening diagnostics, "
            "not rolling-origin forecast validation."
        ),
        "baseline_screening_naive_mae": naive_mae,
        "baseline_screening_seasonal_naive_mae": seasonal_naive_mae,
        "baseline_screening_seasonal_gain": seasonal_gain,
    }, pd.DataFrame(rows)
