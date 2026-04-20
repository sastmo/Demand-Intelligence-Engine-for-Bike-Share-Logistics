from __future__ import annotations

import ctypes.util
import importlib
from importlib import metadata as importlib_metadata
from datetime import datetime
from pathlib import Path
import platform
import sys
from typing import Callable, Iterable


FORECAST_RUNTIME_PACKAGES: tuple[tuple[str, str | None], ...] = (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("statsmodels", "statsmodels"),
    ("torch", "torch"),
    ("lightgbm", "lightgbm"),
    ("xgboost", "xgboost"),
)


def discover_project_root(anchor: str | Path) -> Path:
    resolved = Path(anchor).resolve()
    current = resolved if resolved.is_dir() else resolved.parent
    for candidate in [current, *current.parents]:
        src_root = candidate / "src"
        if all((src_root / package_name).exists() for package_name in ("dashboard", "system_level", "station_level")):
            return candidate
    return current


def emit_summary(title: str, summary: dict[str, object]) -> None:
    print(title, flush=True)
    for key, value in summary.items():
        print(f"{key}: {value}", flush=True)


def emit_notes(notes: Iterable[str], prefix: str = "note") -> None:
    tag = prefix.upper()
    for note in notes:
        print(f"{tag}: {note}", flush=True)


def emit_report(title: str, rows: dict[str, object]) -> None:
    print(title, flush=True)
    for key, value in rows.items():
        print(f"{key}: {value}", flush=True)


def emit_package_report(rows: list[dict[str, object]]) -> None:
    print("package_checks:", flush=True)
    for row in rows:
        detail = f" ({row['detail']})" if row.get("detail") else ""
        print(
            f"- {row['package']}: installed={row['installed']} importable={row['importable']} "
            f"version={row['version']}{detail}",
            flush=True,
        )


def emit_model_report(rows: list[dict[str, object]]) -> None:
    print("model_runtime:", flush=True)
    for row in rows:
        note = f" note={row['note']}" if row.get("note") else ""
        print(
            f"- {row['model_name']}: family={row['family']} implementation={row['implementation']} "
            f"experimental={row['experimental']} tuning_strategy={row['tuning_strategy']}"
            f"{note}",
            flush=True,
        )


def _macos_libomp_status() -> tuple[bool | None, str]:
    if platform.system() != "Darwin":
        return None, "not_applicable"

    candidate_paths = [
        Path("/opt/homebrew/opt/libomp/lib/libomp.dylib"),
        Path("/usr/local/opt/libomp/lib/libomp.dylib"),
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return True, str(candidate)

    discovered = ctypes.util.find_library("omp")
    if discovered:
        return True, str(discovered)
    return False, "missing"


def runtime_environment_report() -> dict[str, object]:
    cfg_path = Path(sys.prefix) / "pyvenv.cfg"
    include_system_site_packages = None
    if cfg_path.exists():
        content = cfg_path.read_text(encoding="utf-8")
        include_system_site_packages = "include-system-site-packages = true" in content

    libomp_available, libomp_detail = _macos_libomp_status()
    return {
        "python_version": sys.version.split()[0],
        "python_supported": sys.version_info >= (3, 10),
        "platform": platform.platform(),
        "virtualenv": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
        "include_system_site_packages": include_system_site_packages,
        "libomp_available": libomp_available,
        "libomp_detail": libomp_detail,
    }


def runtime_environment_notes() -> list[str]:
    notes: list[str] = []
    if sys.version_info < (3, 10):
        notes.append(f"Python {sys.version.split()[0]} is unsupported for this project. Use Python >= 3.10.")

    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        cfg_path = Path(sys.prefix) / "pyvenv.cfg"
        if cfg_path.exists():
            content = cfg_path.read_text(encoding="utf-8")
            if "include-system-site-packages = true" in content:
                notes.append(
                    "This virtual environment includes system site-packages, which can leak incompatible global packages "
                    "into the run. Prefer an isolated venv with include-system-site-packages = false."
                )
    libomp_available, _ = _macos_libomp_status()
    if libomp_available is False:
        notes.append(
            "The macOS OpenMP runtime (`libomp`) is not installed. Native `lightgbm` and `xgboost` backends may fail "
            "or fall back to slower sklearn implementations. Install it with `brew install libomp`."
        )
    return notes


def package_runtime_report(packages: Iterable[tuple[str, str | None]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for module_name, distribution_name in packages:
        package_name = distribution_name or module_name
        installed = True
        try:
            version = importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            installed = False
            version = "missing"

        importable = True
        detail = ""
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - depends on local environment
            importable = False
            detail = f"{type(exc).__name__}: {exc}"

        rows.append(
            {
                "package": module_name,
                "installed": installed,
                "importable": importable,
                "version": version,
                "detail": detail,
            }
        )
    return rows


def default_forecast_package_report() -> list[dict[str, object]]:
    return package_runtime_report(FORECAST_RUNTIME_PACKAGES)


def noop_progress(_: str) -> None:
    return None


def make_progress_logger(enabled: bool = False, prefix: str | None = None) -> Callable[[str], None]:
    if not enabled:
        return noop_progress

    label = f"{prefix.strip()} " if prefix else ""

    def _log(message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {label}{message}", flush=True)

    return _log
