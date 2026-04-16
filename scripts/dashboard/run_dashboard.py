from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metro_bike_share_forecasting.dashboard.app import run_dashboard


if __name__ == "__main__":
    run_dashboard()
