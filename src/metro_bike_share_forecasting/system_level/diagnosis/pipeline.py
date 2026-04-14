from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from metro_bike_share_forecasting.diagnostics.time_series import (
    TimeSeriesDiagnosticsConfig,
    run_time_series_diagnostics,
)


@dataclass(frozen=True)
class SystemLevelDiagnosisConfig:
    output_dir: Path
    series_name: str = "system_level_series"
    frequency: str = "daily"
    timestamp_col: str = "bucket_start"
    value_col: str = "trip_count"
    expected_frequency: str | None = None
    candidate_periods: tuple[int, ...] = ()
    primary_period: int | None = None
    max_acf_lags: int | None = None
    rolling_window: int | None = None
    outlier_threshold: float = 5.0


def run_system_level_diagnostics(frame: pd.DataFrame, config: SystemLevelDiagnosisConfig) -> dict[str, object]:
    return run_time_series_diagnostics(
        frame,
        TimeSeriesDiagnosticsConfig(
            output_dir=Path(config.output_dir),
            series_name=config.series_name,
            frequency=config.frequency,
            timestamp_col=config.timestamp_col,
            value_col=config.value_col,
            expected_frequency=config.expected_frequency,
            candidate_periods=config.candidate_periods,
            primary_period=config.primary_period,
            max_acf_lags=config.max_acf_lags,
            rolling_window=config.rolling_window,
            outlier_threshold=config.outlier_threshold,
        ),
    )

