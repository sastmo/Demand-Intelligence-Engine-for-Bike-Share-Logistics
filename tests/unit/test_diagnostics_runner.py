from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class DiagnosticsRunnerTests(unittest.TestCase):
    def test_runner_executes_end_to_end_on_synthetic_series(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "diagnostics_tmp"
            command = [
                sys.executable,
                str(ROOT / "run_diagnostics.py"),
                "--synthetic-demo",
                "--target-col",
                "value",
                "--frequency",
                "daily",
                "--output-root",
                str(output_root),
            ]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Forecasting diagnostics completed.", result.stdout)
            self.assertTrue((output_root / "figures" / "series.png").exists())
            self.assertTrue((output_root / "tables" / "diagnostics_summary.json").exists())
            self.assertTrue((output_root / "report" / "diagnostics_report.md").exists())


if __name__ == "__main__":
    unittest.main()
