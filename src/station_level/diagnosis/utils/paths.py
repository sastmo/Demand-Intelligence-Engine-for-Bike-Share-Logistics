from __future__ import annotations

from pathlib import Path

from station_level.diagnosis.config import StationDiagnosisConfig


def ensure_analysis_directories(config: StationDiagnosisConfig) -> dict[str, Path]:
    diagnosis_root = Path("diagnosis")
    diagnosis_root.mkdir(parents=True, exist_ok=True)

    legacy_temp = Path("temp")
    system_level_dir = diagnosis_root / "system_level"
    if legacy_temp.exists() and not system_level_dir.exists():
        legacy_temp.rename(system_level_dir)
    system_level_dir.mkdir(parents=True, exist_ok=True)

    station_root = diagnosis_root / "station_level"
    station_root.mkdir(parents=True, exist_ok=True)
    output_root = config.output_root
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    for path in [output_root, tables_dir, figures_dir]:
        path.mkdir(parents=True, exist_ok=True)

    return {
        "diagnosis_root": diagnosis_root,
        "system_level_root": system_level_dir,
        "station_level_root": station_root,
        "output_root": output_root,
        "tables": tables_dir,
        "figures": figures_dir,
    }
