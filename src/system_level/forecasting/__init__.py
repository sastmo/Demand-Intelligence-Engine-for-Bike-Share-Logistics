"""System-level forecasting package.

This module intentionally avoids eager pipeline imports so commands that only
need config loading do not also load optional model libraries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from system_level.forecasting.config import SystemLevelConfig

__all__ = ["SystemLevelConfig", "load_system_level_config", "run_system_level_pipeline"]


def __getattr__(name: str) -> Any:
    if name == "SystemLevelConfig":
        from system_level.forecasting.config import SystemLevelConfig

        return SystemLevelConfig
    raise AttributeError(f"module 'system_level.forecasting' has no attribute {name!r}")


def load_system_level_config(*args, **kwargs):
    from system_level.forecasting.config import load_system_level_config as _load_system_level_config

    return _load_system_level_config(*args, **kwargs)


def run_system_level_pipeline(*args, **kwargs):
    from system_level.forecasting.pipeline import run_system_level_pipeline as _run_system_level_pipeline

    return _run_system_level_pipeline(*args, **kwargs)
