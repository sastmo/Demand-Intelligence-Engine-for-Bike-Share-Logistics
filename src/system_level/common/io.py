from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


OUTPUT_SUBDIRECTORIES = (
    "forecasts",
    "metrics",
    "backtests",
    "models",
    "feature_artifacts",
    "figures",
)


def ensure_output_directories(output_root: Path) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    directories = {name: output_root / name for name in OUTPUT_SUBDIRECTORIES}
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def write_dataframe(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def write_json(payload: dict[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def write_text(content: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
