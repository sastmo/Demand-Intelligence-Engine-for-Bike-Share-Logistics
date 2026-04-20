from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from system_level.diagnosis.types import DiagnosticEvent


FREQUENCY_DEFAULTS: dict[str, dict[str, object]] = {
    "hourly": {
        "expected_frequency": "h",
        "candidate_periods": (24, 168),
        "primary_period": 24,
        "rolling_window": 24,
        "max_acf_lags": 168,
    },
    "daily": {
        "expected_frequency": "D",
        "candidate_periods": (7, 30, 365),
        "primary_period": 7,
        "rolling_window": 28,
        "max_acf_lags": 56,
    },
    "weekly": {
        "expected_frequency": "W-MON",
        "candidate_periods": (4, 13, 52),
        "primary_period": 13,
        "rolling_window": 13,
        "max_acf_lags": 52,
    },
    "monthly": {
        "expected_frequency": "MS",
        "candidate_periods": (12,),
        "primary_period": 12,
        "rolling_window": 12,
        "max_acf_lags": 24,
    },
    "quarterly": {
        "expected_frequency": "QS",
        "candidate_periods": (4,),
        "primary_period": 4,
        "rolling_window": 8,
        "max_acf_lags": 16,
    },
}


def base_frequency_label(frequency: str | None) -> str | None:
    if frequency is None:
        return None
    return str(frequency).split("__", 1)[0].strip().lower()


@dataclass
class DiagnosticConfig:
    series_name: str
    target_col: str
    time_col: str | None = None
    frequency: str | None = None
    expected_frequency: str | None = None
    candidate_periods: tuple[int, ...] = ()
    primary_period: int | None = None
    max_acf_lags: int | None = None
    rolling_window: int | None = None
    outlier_threshold: float = 5.0
    imputation_method: str = "linear_interpolate_then_edge_fill"
    anomaly_method: str = "retrospective_centered_mad"
    output_root: Path = Path("diagnosis/system_level/outputs")
    clean_output: bool = False
    events: tuple[DiagnosticEvent, ...] = field(default_factory=tuple)

    @property
    def resolved_frequency(self) -> str | None:
        return base_frequency_label(self.frequency)


def apply_frequency_defaults(config: DiagnosticConfig) -> DiagnosticConfig:
    defaults = FREQUENCY_DEFAULTS.get(base_frequency_label(config.frequency), {})
    return DiagnosticConfig(
        series_name=config.series_name,
        target_col=config.target_col,
        time_col=config.time_col,
        frequency=config.frequency,
        expected_frequency=config.expected_frequency or defaults.get("expected_frequency"),
        candidate_periods=config.candidate_periods or tuple(defaults.get("candidate_periods", ())),
        primary_period=config.primary_period or defaults.get("primary_period"),
        max_acf_lags=config.max_acf_lags or defaults.get("max_acf_lags"),
        rolling_window=config.rolling_window or defaults.get("rolling_window"),
        outlier_threshold=config.outlier_threshold,
        imputation_method=config.imputation_method,
        anomaly_method=config.anomaly_method,
        output_root=config.output_root,
        clean_output=config.clean_output,
        events=config.events,
    )
