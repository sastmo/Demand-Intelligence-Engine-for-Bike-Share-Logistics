from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class StationLevelForecastConfig:
    project_root: Path
    daily_aggregate_path: Path
    diagnosis_summary_path: Path | None
    date_column: str
    station_column: str
    target_column: str
    segment_type: str
    in_service_column: str
    forecast_horizons: tuple[int, ...]
    extended_horizon: int
    lags: tuple[int, ...]
    rolling_windows: tuple[int, ...]
    holiday_country: str
    include_category_feature: bool
    include_cluster_feature: bool
    initial_train_size: int
    step_size: int
    max_folds: int
    recent_activity_window_days: int
    min_recent_service_days: int
    baselines_enabled: dict[str, bool]
    tree_enabled: dict[str, bool]
    deepar_enabled: bool
    random_state: int
    tune_enabled: bool
    deepar_context_length: int
    deepar_hidden_size: int
    deepar_embedding_dim: int
    deepar_batch_size: int
    deepar_epochs: int
    deepar_learning_rate: float
    output_root: Path

    @property
    def production_horizons(self) -> tuple[int, ...]:
        values = list(self.forecast_horizons)
        if self.extended_horizon not in values:
            values.append(self.extended_horizon)
        return tuple(sorted(values))

    @property
    def max_backtest_horizon(self) -> int:
        return max(self.forecast_horizons)

    @property
    def enabled_model_keys(self) -> list[str]:
        keys: list[str] = []
        keys.extend([name for name, enabled in self.baselines_enabled.items() if enabled])
        keys.extend([name for name, enabled in self.tree_enabled.items() if enabled])
        if self.deepar_enabled:
            keys.append("deepar")
        return keys


def _resolve_path(project_root: Path, candidate: str | None) -> Path | None:
    if not candidate:
        return None
    path = Path(candidate)
    return path if path.is_absolute() else project_root / path


def load_station_level_config(config_path: str | Path) -> StationLevelForecastConfig:
    config_path = Path(config_path)
    project_root = config_path.resolve().parents[2]
    payload = yaml.safe_load(config_path.read_text())

    input_cfg = payload["input"]
    feature_cfg = payload["features"]
    backtest_cfg = payload["backtest"]
    model_cfg = payload["models"]
    training_cfg = payload["training"]
    output_cfg = payload["output"]

    return StationLevelForecastConfig(
        project_root=project_root,
        daily_aggregate_path=_resolve_path(project_root, input_cfg["daily_aggregate_path"]),
        diagnosis_summary_path=_resolve_path(project_root, input_cfg.get("diagnosis_summary_path")),
        date_column=input_cfg["date_column"],
        station_column=str(input_cfg["station_column"]),
        target_column=input_cfg["target_column"],
        segment_type=str(input_cfg["segment_type"]),
        in_service_column=input_cfg["in_service_column"],
        forecast_horizons=tuple(int(value) for value in payload["forecast"]["main_horizons"]),
        extended_horizon=int(payload["forecast"]["extended_horizon"]),
        lags=tuple(int(value) for value in feature_cfg["lags"]),
        rolling_windows=tuple(int(value) for value in feature_cfg["rolling_windows"]),
        holiday_country=str(feature_cfg["holiday_country"]),
        include_category_feature=bool(feature_cfg.get("include_category_feature", False)),
        include_cluster_feature=bool(feature_cfg.get("include_cluster_feature", False)),
        initial_train_size=int(backtest_cfg["initial_train_size"]),
        step_size=int(backtest_cfg["step_size"]),
        max_folds=int(backtest_cfg["max_folds"]),
        recent_activity_window_days=int(backtest_cfg.get("recent_activity_window_days", 90)),
        min_recent_service_days=int(backtest_cfg.get("min_recent_service_days", 3)),
        baselines_enabled={str(key): bool(value) for key, value in model_cfg["baselines"].items()},
        tree_enabled={str(key): bool(value) for key, value in model_cfg["tree"].items()},
        deepar_enabled=bool(model_cfg["deep"].get("deepar", False)),
        random_state=int(training_cfg.get("random_state", 42)),
        tune_enabled=bool(training_cfg.get("tune_enabled", False)),
        deepar_context_length=int(training_cfg.get("deepar_context_length", 28)),
        deepar_hidden_size=int(training_cfg.get("deepar_hidden_size", 32)),
        deepar_embedding_dim=int(training_cfg.get("deepar_embedding_dim", 16)),
        deepar_batch_size=int(training_cfg.get("deepar_batch_size", 2048)),
        deepar_epochs=int(training_cfg.get("deepar_epochs", 3)),
        deepar_learning_rate=float(training_cfg.get("deepar_learning_rate", 0.001)),
        output_root=_resolve_path(project_root, output_cfg["root"]),
    )
