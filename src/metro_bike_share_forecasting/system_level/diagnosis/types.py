from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DiagnosticEvent:
    label: str
    timestamp: pd.Timestamp
    color: str = "#d9534f"


@dataclass
class DiagnosticResult:
    summary: dict[str, Any]
    output_root: Path
    figures_dir: Path
    tables_dir: Path
    report_dir: Path | None = None
    figures: dict[str, Path] = field(default_factory=dict)
    tables: dict[str, Path] = field(default_factory=dict)
    report_path: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "output_root": str(self.output_root),
            "figures_dir": str(self.figures_dir),
            "tables_dir": str(self.tables_dir),
            "report_dir": str(self.report_dir) if self.report_dir else None,
            "figures": {name: str(path) for name, path in self.figures.items()},
            "tables": {name: str(path) for name, path in self.tables.items()},
            "report_path": str(self.report_path) if self.report_path else None,
        }
