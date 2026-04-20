from __future__ import annotations

import unittest
from pathlib import Path

from dashboard.editor import asset_catalog, editor_storage_path, slugify


class DashboardEditorTests(unittest.TestCase):
    def test_slugify_normalizes_text(self) -> None:
        self.assertEqual(slugify("System-Level Forecasting"), "system-level-forecasting")

    def test_editor_storage_path_is_page_specific(self) -> None:
        path = editor_storage_path("system_forecast")
        self.assertEqual(path.name, "system_forecast.json")
        self.assertIn("src/dashboard/content/pages", path.as_posix())

    def test_asset_catalog_includes_expected_figure_groups(self) -> None:
        catalog = asset_catalog()
        self.assertIn("Diagnosis / System", catalog)
        self.assertIn("Diagnosis / Station", catalog)
        self.assertIn("Forecasts / System", catalog)
        self.assertIn("Forecasts / Station", catalog)
        self.assertIn("diagnosis/system_level/outputs/figures/series.png", catalog["Diagnosis / System"])

    def test_dashboard_source_uses_width_api_instead_of_deprecated_container_flag(self) -> None:
        dashboard_root = Path(__file__).resolve().parents[2] / "src" / "dashboard"
        source = "\n".join(path.read_text(encoding="utf-8") for path in dashboard_root.rglob("*.py"))
        self.assertNotIn("use_container_width", source)

    def test_figure_asset_state_is_not_rewritten_after_widget_binding(self) -> None:
        editor_source = (Path(__file__).resolve().parents[2] / "src" / "dashboard" / "editor.py").read_text(encoding="utf-8")
        self.assertNotIn('st.session_state[self._state_key(block_id, "asset_path")] = selected_asset', editor_source)


if __name__ == "__main__":
    unittest.main()
