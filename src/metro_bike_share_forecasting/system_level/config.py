from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SystemLevelConfig:
    project_root: Path
    daily_aggregate_path: Path
    cleaned_trip_path: Path
    external_features_path: Path | None
    external_date_column: str
    target_column: str
    date_column: str
    frequency: str
    segment_type: str
    segment_id: str
    forecast_horizons: tuple[int, ...]
    extended_horizon: int
    lags: tuple[int, ...]
    rolling_windows: tuple[int, ...]
    holiday_country: str
    include_weekly_fourier: bool
    include_yearly_fourier: bool
    weekly_fourier_order: int
    yearly_fourier_order: int
    initial_train_size: int
    step_size: int
    max_folds: int
    baselines_enabled: dict[str, bool]
    classical_enabled: dict[str, bool]
    ml_enabled: dict[str, bool]
    output_root: Path

    @property
    def production_horizons(self) -> tuple[int, ...]:
        values = list(self.forecast_horizons)
        if self.extended_horizon not in values:
            values.append(self.extended_horizon)
        return tuple(sorted(values))

    @property
    def enabled_model_keys(self) -> list[str]:
        ordered: list[str] = []
        for family in (self.baselines_enabled, self.classical_enabled, self.ml_enabled):
            ordered.extend([name for name, enabled in family.items() if enabled])
        return ordered


def _resolve_path(project_root: Path, candidate: str | None) -> Path | None:
    if not candidate:
        return None
    path = Path(candidate)
    return path if path.is_absolute() else project_root / path


def load_system_level_config(config_path: str | Path) -> SystemLevelConfig:
    config_path = Path(config_path)
    project_root = config_path.resolve().parents[2]
    payload = yaml.safe_load(config_path.read_text())

    input_cfg = payload["input"]
    feature_cfg = payload["features"]
    backtest_cfg = payload["backtest"]
    model_cfg = payload["models"]
    output_cfg = payload["output"]

    return SystemLevelConfig(
        project_root=project_root,
        daily_aggregate_path=_resolve_path(project_root, input_cfg["daily_aggregate_path"]),
        cleaned_trip_path=_resolve_path(project_root, input_cfg["cleaned_trip_path"]),
        external_features_path=_resolve_path(project_root, feature_cfg.get("external_features_path")),
        external_date_column=feature_cfg.get("external_date_column", "date"),
        target_column=input_cfg["target_column"],
        date_column=input_cfg["date_column"],
        frequency=input_cfg["frequency"],
        segment_type=str(input_cfg["segment_type"]),
        segment_id=str(input_cfg["segment_id"]),
        forecast_horizons=tuple(int(value) for value in payload["forecast"]["main_horizons"]),
        extended_horizon=int(payload["forecast"]["extended_horizon"]),
        lags=tuple(int(value) for value in feature_cfg["lags"]),
        rolling_windows=tuple(int(value) for value in feature_cfg["rolling_windows"]),
        holiday_country=str(feature_cfg["holiday_country"]),
        include_weekly_fourier=bool(feature_cfg.get("include_weekly_fourier", True)),
        include_yearly_fourier=bool(feature_cfg.get("include_yearly_fourier", True)),
        weekly_fourier_order=int(feature_cfg.get("weekly_fourier_order", 3)),
        yearly_fourier_order=int(feature_cfg.get("yearly_fourier_order", 3)),
        initial_train_size=int(backtest_cfg["initial_train_size"]),
        step_size=int(backtest_cfg["step_size"]),
        max_folds=int(backtest_cfg["max_folds"]),
        baselines_enabled={str(key): bool(value) for key, value in model_cfg["baselines"].items()},
        classical_enabled={str(key): bool(value) for key, value in model_cfg["classical"].items()},
        ml_enabled={str(key): bool(value) for key, value in model_cfg["ml"].items()},
        output_root=_resolve_path(project_root, output_cfg["root"]),
    )
