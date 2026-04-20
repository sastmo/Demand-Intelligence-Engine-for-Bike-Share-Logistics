from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import streamlit as st

from dashboard.data import repo_root

DEFAULT_FONT_FAMILY = '"Avenir Next", "Segoe UI", sans-serif'
FONT_OPTIONS: dict[str, str] = {
    "Avenir Sans": DEFAULT_FONT_FAMILY,
    "System Sans": 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    "Georgia Serif": 'Georgia, "Times New Roman", serif',
    "IBM Plex Sans": '"IBM Plex Sans", "Segoe UI", sans-serif',
    "Source Serif": '"Source Serif 4", Georgia, serif',
}
DEFAULT_STYLE = {
    "title_color": "#e9eef5",
    "body_color": "#9aacbe",
    "font_family": DEFAULT_FONT_FAMILY,
    "font_scale": 1.0,
}
ASSET_GROUPS = {
    "Diagnosis / System": Path("diagnosis/system_level/outputs/figures"),
    "Diagnosis / Station": Path("diagnosis/station_level/outputs/figures"),
    "Forecasts / System": Path("forecasts/system_level/figures"),
    "Forecasts / Station": Path("forecasts/station_level/figures"),
}


def slugify(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return text or "block"


def editor_storage_dir() -> Path:
    return Path(__file__).resolve().parent / "content" / "pages"


def editor_storage_path(page_key: str) -> Path:
    return editor_storage_dir() / f"{page_key}.json"


@lru_cache(maxsize=1)
def asset_catalog() -> dict[str, list[str]]:
    root = repo_root()
    catalog: dict[str, list[str]] = {}
    for label, relative_dir in ASSET_GROUPS.items():
        directory = root / relative_dir
        paths = sorted(
            str(path.relative_to(root))
            for path in directory.glob("*")
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}
        )
        catalog[label] = paths
    return catalog


def default_style(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    style = dict(DEFAULT_STYLE)
    if overrides:
        style.update({key: value for key, value in overrides.items() if value is not None})
    return style


def _deep_merge(default: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(default)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


@dataclass
class PageBlock:
    block_type: str
    block_id: str
    label: str
    default: dict[str, Any]


class PageContentEditor:
    def __init__(self, page_key: str, page_label: str) -> None:
        self.page_key = page_key
        self.page_label = page_label
        self.storage_path = editor_storage_path(page_key)
        self.saved_overrides = self._load_saved_overrides()
        self.blocks: list[PageBlock] = []
        self._registered_ids: set[str] = set()
        self._block_index: dict[str, PageBlock] = {}

    def _load_saved_overrides(self) -> dict[str, dict[str, Any]]:
        if not self.storage_path.exists():
            return {}
        try:
            payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        if not isinstance(payload, dict):
            return {}
        return {
            str(block_id): dict(block_value)
            for block_id, block_value in payload.items()
            if isinstance(block_value, dict)
        }

    def _register(self, block_type: str, block_id: str, label: str, default: dict[str, Any]) -> None:
        if block_id in self._registered_ids:
            return
        self._registered_ids.add(block_id)
        block = PageBlock(block_type=block_type, block_id=block_id, label=label, default=default)
        self.blocks.append(block)
        self._block_index[block_id] = block

    def _state_key(self, block_id: str, field: str) -> str:
        return f"dashboard_editor:{self.page_key}:{block_id}:{field}"

    def _ensure_state(self, block_id: str, config: dict[str, Any]) -> None:
        field_map = {
            "title": config.get("title", ""),
            "subtitle": config.get("subtitle", ""),
            "body": config.get("body", ""),
            "label": config.get("label", ""),
            "description": config.get("description", ""),
            "tone": config.get("tone", "default"),
            "title_color": config.get("style", {}).get("title_color", DEFAULT_STYLE["title_color"]),
            "body_color": config.get("style", {}).get("body_color", DEFAULT_STYLE["body_color"]),
            "font_family": config.get("style", {}).get("font_family", DEFAULT_STYLE["font_family"]),
            "font_scale": float(config.get("style", {}).get("font_scale", DEFAULT_STYLE["font_scale"])),
            "asset_path": config.get("asset_path", ""),
            "asset_group": self._group_for_asset(config.get("asset_path", "")),
            "bullets": "\n".join(config.get("bullets", [])),
        }
        for field, value in field_map.items():
            key = self._state_key(block_id, field)
            if key not in st.session_state:
                st.session_state[key] = value

    def _style_from_state(self, block_id: str) -> dict[str, Any]:
        return {
            "title_color": st.session_state.get(self._state_key(block_id, "title_color"), DEFAULT_STYLE["title_color"]),
            "body_color": st.session_state.get(self._state_key(block_id, "body_color"), DEFAULT_STYLE["body_color"]),
            "font_family": st.session_state.get(self._state_key(block_id, "font_family"), DEFAULT_STYLE["font_family"]),
            "font_scale": float(st.session_state.get(self._state_key(block_id, "font_scale"), DEFAULT_STYLE["font_scale"])),
        }

    def _current_from_state(self, block_type: str, block_id: str, default: dict[str, Any]) -> dict[str, Any]:
        self._ensure_state(block_id, default)
        if block_type == "bullet_box":
            bullets_text = st.session_state.get(self._state_key(block_id, "bullets"), "")
            bullets = [line.strip() for line in str(bullets_text).splitlines() if line.strip()]
            return {
                "title": st.session_state.get(self._state_key(block_id, "title"), default.get("title", "")),
                "bullets": bullets,
                "tone": st.session_state.get(self._state_key(block_id, "tone"), default.get("tone", "default")),
                "style": self._style_from_state(block_id),
            }
        if block_type == "story_box":
            return {
                "title": st.session_state.get(self._state_key(block_id, "title"), default.get("title", "")),
                "body": st.session_state.get(self._state_key(block_id, "body"), default.get("body", "")),
                "tone": st.session_state.get(self._state_key(block_id, "tone"), default.get("tone", "default")),
                "style": self._style_from_state(block_id),
            }
        if block_type == "section_header":
            return {
                "title": st.session_state.get(self._state_key(block_id, "title"), default.get("title", "")),
                "subtitle": st.session_state.get(self._state_key(block_id, "subtitle"), default.get("subtitle", "")),
                "style": self._style_from_state(block_id),
            }
        if block_type == "figure":
            return {
                "label": st.session_state.get(self._state_key(block_id, "label"), default.get("label", "")),
                "description": st.session_state.get(self._state_key(block_id, "description"), default.get("description", "")),
                "asset_path": st.session_state.get(self._state_key(block_id, "asset_path"), default.get("asset_path", "")),
                "style": self._style_from_state(block_id),
            }
        return default

    def _prepare_block(self, block_type: str, block_id: str, label: str, default: dict[str, Any]) -> dict[str, Any]:
        saved = self.saved_overrides.get(block_id, {})
        merged = _deep_merge(default, saved)
        self._register(block_type, block_id, label, default)
        return self._current_from_state(block_type, block_id, merged)

    def bullet_box(self, block_id: str, *, title: str, bullets: list[str], tone: str = "default", style: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._prepare_block(
            "bullet_box",
            block_id,
            title,
            {"title": title, "bullets": bullets, "tone": tone, "style": default_style(style)},
        )

    def story_box(self, block_id: str, *, title: str, body: str, tone: str = "default", style: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._prepare_block(
            "story_box",
            block_id,
            title,
            {"title": title, "body": body, "tone": tone, "style": default_style(style)},
        )

    def section_header(self, block_id: str, *, title: str, subtitle: str | None = None, style: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._prepare_block(
            "section_header",
            block_id,
            title,
            {"title": title, "subtitle": subtitle or "", "style": default_style(style)},
        )

    def figure_block(
        self,
        block_id: str,
        *,
        label: str,
        description: str,
        asset_path: str = "",
        style: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._prepare_block(
            "figure",
            block_id,
            label,
            {
                "label": label,
                "description": description,
                "asset_path": asset_path,
                "style": default_style(style),
            },
        )

    def _group_for_asset(self, asset_path: str) -> str:
        if not asset_path:
            return next(iter(ASSET_GROUPS))
        for group_label, prefix in ASSET_GROUPS.items():
            if str(asset_path).startswith(str(prefix).replace("\\", "/")):
                return group_label
        return next(iter(ASSET_GROUPS))

    def _style_payload(self, block_id: str) -> dict[str, Any]:
        return self._style_from_state(block_id)

    def _block_payload(self, block: PageBlock) -> dict[str, Any]:
        block_id = block.block_id
        payload: dict[str, Any]
        if block.block_type == "bullet_box":
            bullets_text = st.session_state.get(self._state_key(block_id, "bullets"), "")
            payload = {
                "title": st.session_state.get(self._state_key(block_id, "title"), block.default.get("title", "")),
                "bullets": [line.strip() for line in str(bullets_text).splitlines() if line.strip()],
                "tone": st.session_state.get(self._state_key(block_id, "tone"), block.default.get("tone", "default")),
            }
        elif block.block_type == "story_box":
            payload = {
                "title": st.session_state.get(self._state_key(block_id, "title"), block.default.get("title", "")),
                "body": st.session_state.get(self._state_key(block_id, "body"), block.default.get("body", "")),
                "tone": st.session_state.get(self._state_key(block_id, "tone"), block.default.get("tone", "default")),
            }
        elif block.block_type == "section_header":
            payload = {
                "title": st.session_state.get(self._state_key(block_id, "title"), block.default.get("title", "")),
                "subtitle": st.session_state.get(self._state_key(block_id, "subtitle"), block.default.get("subtitle", "")),
            }
        elif block.block_type == "figure":
            payload = {
                "label": st.session_state.get(self._state_key(block_id, "label"), block.default.get("label", "")),
                "description": st.session_state.get(self._state_key(block_id, "description"), block.default.get("description", "")),
                "asset_path": st.session_state.get(self._state_key(block_id, "asset_path"), block.default.get("asset_path", "")),
            }
        else:
            payload = {}
        payload["style"] = self._style_payload(block_id)
        return payload

    def save(self) -> None:
        payload = {block.block_id: self._block_payload(block) for block in self.blocks}
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        self.saved_overrides = payload

    def reset(self) -> None:
        if self.storage_path.exists():
            self.storage_path.unlink()
        self.saved_overrides = {}
        for block in self.blocks:
            defaults = block.default
            if block.block_type == "bullet_box":
                st.session_state[self._state_key(block.block_id, "title")] = defaults["title"]
                st.session_state[self._state_key(block.block_id, "bullets")] = "\n".join(defaults["bullets"])
                st.session_state[self._state_key(block.block_id, "tone")] = defaults["tone"]
            elif block.block_type == "story_box":
                st.session_state[self._state_key(block.block_id, "title")] = defaults["title"]
                st.session_state[self._state_key(block.block_id, "body")] = defaults["body"]
                st.session_state[self._state_key(block.block_id, "tone")] = defaults["tone"]
            elif block.block_type == "section_header":
                st.session_state[self._state_key(block.block_id, "title")] = defaults["title"]
                st.session_state[self._state_key(block.block_id, "subtitle")] = defaults["subtitle"]
            elif block.block_type == "figure":
                st.session_state[self._state_key(block.block_id, "label")] = defaults["label"]
                st.session_state[self._state_key(block.block_id, "description")] = defaults["description"]
                st.session_state[self._state_key(block.block_id, "asset_path")] = defaults["asset_path"]
                st.session_state[self._state_key(block.block_id, "asset_group")] = self._group_for_asset(defaults["asset_path"])
            current_font_family = defaults.get("style", default_style()).get("font_family", DEFAULT_FONT_FAMILY)
            for label, family in FONT_OPTIONS.items():
                if family == current_font_family:
                    st.session_state[self._state_key(block.block_id, "font_family_label")] = label
                    break
            for style_key, style_value in block.default.get("style", default_style()).items():
                st.session_state[self._state_key(block.block_id, style_key)] = style_value

    def _render_style_controls(self, block_id: str) -> None:
        st.color_picker("Title color", key=self._state_key(block_id, "title_color"))
        st.color_picker("Body color", key=self._state_key(block_id, "body_color"))
        current_family = st.session_state.get(self._state_key(block_id, "font_family"), DEFAULT_FONT_FAMILY)
        labels = list(FONT_OPTIONS)
        current_index = 0
        for index, label in enumerate(labels):
            if FONT_OPTIONS[label] == current_family:
                current_index = index
                break
        selected_label = st.selectbox("Font family", labels, index=current_index, key=self._state_key(block_id, "font_family_label"))
        st.session_state[self._state_key(block_id, "font_family")] = FONT_OPTIONS[selected_label]
        st.slider("Font scale", min_value=0.8, max_value=1.4, value=float(st.session_state.get(self._state_key(block_id, "font_scale"), 1.0)), step=0.05, key=self._state_key(block_id, "font_scale"))

    def enabled(self) -> bool:
        return bool(st.session_state.get(f"dashboard_editor_toggle:{self.page_key}", False))

    def _render_block_controls(self, block: PageBlock) -> None:
        block_id = block.block_id
        if block.block_type == "bullet_box":
            st.text_input("Title", key=self._state_key(block_id, "title"))
            st.selectbox("Tone", ["default", "accent"], key=self._state_key(block_id, "tone"))
            st.text_area("Bullets", key=self._state_key(block_id, "bullets"), height=160, help="One bullet per line.")
            self._render_style_controls(block_id)
        elif block.block_type == "story_box":
            st.text_input("Title", key=self._state_key(block_id, "title"))
            st.selectbox("Tone", ["default", "accent"], key=self._state_key(block_id, "tone"))
            st.text_area("Body", key=self._state_key(block_id, "body"), height=180)
            self._render_style_controls(block_id)
        elif block.block_type == "section_header":
            st.text_input("Title", key=self._state_key(block_id, "title"))
            st.text_input("Subtitle", key=self._state_key(block_id, "subtitle"))
            self._render_style_controls(block_id)
        elif block.block_type == "figure":
            catalog = asset_catalog()
            current_group = st.session_state.get(self._state_key(block_id, "asset_group"), self._group_for_asset(st.session_state.get(self._state_key(block_id, "asset_path"), "")))
            groups = list(catalog)
            group_index = groups.index(current_group) if current_group in groups else 0
            selected_group = st.selectbox("Asset folder", groups, index=group_index, key=self._state_key(block_id, "asset_group"))
            options = catalog.get(selected_group, [])
            current_asset = st.session_state.get(self._state_key(block_id, "asset_path"), block.default.get("asset_path", ""))
            if options and current_asset not in options:
                current_asset = options[0]
                st.session_state[self._state_key(block_id, "asset_path")] = current_asset
            asset_index = options.index(current_asset) if options and current_asset in options else 0
            if options:
                st.selectbox("Figure asset", options, index=asset_index, key=self._state_key(block_id, "asset_path"))
            else:
                st.caption("No figure files found in this asset folder.")
            st.text_input("Label", key=self._state_key(block_id, "label"))
            st.text_area("Description", key=self._state_key(block_id, "description"), height=180)
            self._render_style_controls(block_id)

    def render_inline_handle(self, block_id: str, *, label: str = "Edit") -> None:
        if not self.enabled():
            return
        block = self._block_index.get(block_id)
        if block is None:
            return
        with st.popover(label, width="content"):
            st.caption(block.label)
            self._render_block_controls(block)

    def render_sidebar(self) -> None:
        st.sidebar.markdown("## Page Editor")
        enabled = st.sidebar.toggle("Enable page editor", key=f"dashboard_editor_toggle:{self.page_key}")
        if not enabled:
            return
        st.sidebar.caption(f"Saves to `{self.storage_path.relative_to(repo_root())}`")
        st.sidebar.caption("Use the inline Edit controls on the page to update section headers, narrative boxes, and image cards. Data tables and live charts stay code-driven.")
        save_col, reset_col = st.sidebar.columns(2)
        if save_col.button("Save edits", key=f"save_dashboard_page:{self.page_key}", width="stretch"):
            self.save()
            st.sidebar.success("Page edits saved.")
        if reset_col.button("Reset page", key=f"reset_dashboard_page:{self.page_key}", width="stretch"):
            self.reset()
            st.sidebar.info("Saved overrides cleared for this page.")
