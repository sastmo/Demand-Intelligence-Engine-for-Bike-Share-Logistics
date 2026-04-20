"""Station-level forecasting package.

Keep imports lightweight so configuration and CLI setup do not eagerly require
optional ML backends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from station_level.forecasting.config import StationLevelForecastConfig

__all__ = ["StationLevelForecastConfig", "load_station_level_config", "run_station_level_pipeline"]


def __getattr__(name: str) -> Any:
    if name == "StationLevelForecastConfig":
        from station_level.forecasting.config import StationLevelForecastConfig

        return StationLevelForecastConfig
    raise AttributeError(f"module 'station_level.forecasting' has no attribute {name!r}")


def load_station_level_config(*args, **kwargs):
    from station_level.forecasting.config import load_station_level_config as _load_station_level_config

    return _load_station_level_config(*args, **kwargs)


def run_station_level_pipeline(*args, **kwargs):
    from station_level.forecasting.pipeline import run_station_level_pipeline as _run_station_level_pipeline

    return _run_station_level_pipeline(*args, **kwargs)
