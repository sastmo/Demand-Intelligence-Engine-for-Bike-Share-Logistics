from __future__ import annotations

from pathlib import Path

from metro_bike_share_forecasting.station_level.diagnosis.config import StationDiagnosisConfig


def ensure_analysis_directories(config: StationDiagnosisConfig) -> dict[str, Path]:
    """Ensure diagnosis folders exist and rename a legacy temp folder if present."""

    diagnosis_root = Path("diagnosis")
    diagnosis_root.mkdir(parents=True, exist_ok=True)

    legacy_temp = Path("temp")
    system_level_dir = diagnosis_root / "system_level_analysis"
    if legacy_temp.exists() and not system_level_dir.exists():
        legacy_temp.rename(system_level_dir)
    system_level_dir.mkdir(parents=True, exist_ok=True)

    station_root = diagnosis_root / "station_level_analysis"
    station_root.mkdir(parents=True, exist_ok=True)
    (station_root / "assets").mkdir(parents=True, exist_ok=True)
    output_root = config.output_root
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    reports_dir = output_root / "reports"
    diagnostics_dir = output_root / "diagnostics"
    for path in [output_root, tables_dir, figures_dir, reports_dir, diagnostics_dir]:
        path.mkdir(parents=True, exist_ok=True)

    return {
        "diagnosis_root": diagnosis_root,
        "system_level_analysis": system_level_dir,
        "station_level_root": station_root,
        "output_root": output_root,
        "tables": tables_dir,
        "figures": figures_dir,
        "reports": reports_dir,
        "diagnostics": diagnostics_dir,
    }
