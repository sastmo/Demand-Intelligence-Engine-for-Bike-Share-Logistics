from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from forecasting_diagnostics import DiagnosticConfig, DiagnosticEvent, run_forecasting_diagnostics
from metro_bike_share_forecasting.features.regime import RegimeDefinition


@dataclass
class TimeSeriesDiagnosticsConfig:
    output_dir: Path
    series_name: str
    frequency: str
    timestamp_col: str = "timestamp"
    value_col: str = "value"
    expected_frequency: str | None = None
    candidate_periods: tuple[int, ...] = ()
    primary_period: int | None = None
    max_acf_lags: int | None = None
    rolling_window: int | None = None
    outlier_threshold: float = 5.0
    events: tuple[DiagnosticEvent, ...] = field(default_factory=tuple)


def _mirror_for_legacy_consumers(result) -> None:
    legacy_root = Path(result.output_root)
    legacy_root.mkdir(parents=True, exist_ok=True)

    for path in result.figures.values():
        shutil.copy2(path, legacy_root / path.name)

    for name, path in result.tables.items():
        if name.startswith("diagnostics_summary"):
            shutil.copy2(path, legacy_root / path.name)
    for name in ("weekday_profile", "monthly_profile", "intraday_profile"):
        if name in result.tables:
            path = result.tables[name]
            shutil.copy2(path, legacy_root / path.name)
    if result.report_path is not None:
        shutil.copy2(result.report_path, legacy_root / result.report_path.name)


def run_time_series_diagnostics(frame: pd.DataFrame, config: TimeSeriesDiagnosticsConfig) -> dict[str, object]:
    result = run_forecasting_diagnostics(
        frame,
        DiagnosticConfig(
            series_name=config.series_name,
            target_col=config.value_col,
            time_col=config.timestamp_col,
            frequency=config.frequency,
            expected_frequency=config.expected_frequency,
            candidate_periods=config.candidate_periods,
            primary_period=config.primary_period,
            max_acf_lags=config.max_acf_lags,
            rolling_window=config.rolling_window,
            outlier_threshold=config.outlier_threshold,
            output_root=Path(config.output_dir),
            clean_output=True,
            events=config.events,
        ),
    )
    _mirror_for_legacy_consumers(result)
    return result.summary


def run_diagnostics(
    frame: pd.DataFrame,
    frequency: str,
    output_root: Path,
    regime_definition: RegimeDefinition | None = None,
) -> dict[str, object]:
    events: list[DiagnosticEvent] = []
    if regime_definition is not None:
        events.extend(
            [
                DiagnosticEvent("shock", pd.Timestamp(regime_definition.pandemic_shock_start)),
                DiagnosticEvent("recovery", pd.Timestamp(regime_definition.recovery_start)),
                DiagnosticEvent("post", pd.Timestamp(regime_definition.post_pandemic_start)),
            ]
        )

    summary = run_time_series_diagnostics(
        frame,
        TimeSeriesDiagnosticsConfig(
            output_dir=Path(output_root) / frequency,
            series_name=frequency,
            frequency=str(frequency).split("__", 1)[0].strip().lower(),
            timestamp_col="bucket_start",
            value_col="trip_count",
            events=tuple(events),
        ),
    )
    summary["series_key"] = frequency
    return summary
