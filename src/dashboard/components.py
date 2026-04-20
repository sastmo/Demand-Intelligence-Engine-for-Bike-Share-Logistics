from __future__ import annotations

import base64
import html
import mimetypes
from functools import lru_cache
from pathlib import Path
from textwrap import dedent

import pandas as pd
import streamlit as st

from dashboard.data import format_short_number, repo_root
from dashboard.editor import PageContentEditor, default_style, slugify

TOP_HIGHLIGHT_TITLE = "Key takeaways"


def _resolve_block_id(prefix: str, seed: str, explicit: str | None = None) -> str:
    return explicit or f"{prefix}:{slugify(seed)}"


def _inline_text_style(style: dict[str, object] | None, *, base_size: float, color_key: str) -> str:
    style = default_style(style)
    font_scale = float(style.get("font_scale", 1.0))
    declarations = [
        f"color: {style.get(color_key)}",
        f"font-family: {style.get('font_family')}",
        f"font-size: {base_size * font_scale:.2f}rem",
    ]
    return "; ".join(declarations)


def _render_inline_editor_anchor(editor: PageContentEditor | None, block_id: str) -> None:
    if editor is None or not editor.enabled():
        return
    spacer_col, edit_col = st.columns([0.8, 0.2])
    with edit_col:
        editor.render_inline_handle(block_id)


def _render_html(markup: str) -> None:
    st.markdown(dedent(markup).strip(), unsafe_allow_html=True)


def metric_cards(cards: list[tuple[str, str, str]]) -> None:
    if not cards:
        return
    grid = "".join(
        dedent(
            f"""
            <div class="metric-card">
              <div class="metric-label">{html.escape(label)}</div>
              <div class="metric-value">{html.escape(value)}</div>
              <div class="metric-caption">{html.escape(caption)}</div>
            </div>
            """
        ).strip()
        for label, value, caption in cards
    )
    _render_html(
        f"""
        <div class="metric-grid">{grid}</div>
        """
    )


def bullet_box(
    title: str,
    bullets: list[str],
    tone: str = "default",
    *,
    editor: PageContentEditor | None = None,
    block_id: str | None = None,
    style: dict[str, object] | None = None,
) -> None:
    if editor is not None:
        resolved_block_id = _resolve_block_id("bullet-box", title, block_id)
        config = editor.bullet_box(
            resolved_block_id,
            title=title,
            bullets=bullets,
            tone=tone,
            style=style,
        )
        title = str(config["title"])
        bullets = list(config["bullets"])
        tone = str(config["tone"])
        style = dict(config["style"])
        _render_inline_editor_anchor(editor, resolved_block_id)
    title_style = _inline_text_style(style, base_size=0.92, color_key="title_color")
    body_style = _inline_text_style(style, base_size=1.0, color_key="body_color")
    items = "".join(f"<li>{html.escape(str(bullet))}</li>" for bullet in bullets if bullet)
    _render_html(
        f"""
        <div class="insight-shell {tone}">
          <div class="insight-title" style="{title_style}">{html.escape(title)}</div>
          <ul class="insight-list" style="{body_style}">{items}</ul>
        </div>
        """
    )


def story_box(
    title: str,
    text: str,
    tone: str = "default",
    *,
    editor: PageContentEditor | None = None,
    block_id: str | None = None,
    style: dict[str, object] | None = None,
) -> None:
    if editor is not None:
        resolved_block_id = _resolve_block_id("story-box", title, block_id)
        config = editor.story_box(
            resolved_block_id,
            title=title,
            body=text,
            tone=tone,
            style=style,
        )
        title = str(config["title"])
        text = str(config["body"])
        tone = str(config["tone"])
        style = dict(config["style"])
        _render_inline_editor_anchor(editor, resolved_block_id)
    title_style = _inline_text_style(style, base_size=0.92, color_key="title_color")
    body_style = _inline_text_style(style, base_size=0.98, color_key="body_color")
    _render_html(
        f"""
        <div class="story-shell {tone}">
          <div class="insight-title" style="{title_style}">{html.escape(title)}</div>
          <p style="{body_style}">{html.escape(text)}</p>
        </div>
        """
    )


def section_header(
    title: str,
    subtitle: str | None = None,
    *,
    editor: PageContentEditor | None = None,
    block_id: str | None = None,
    style: dict[str, object] | None = None,
) -> None:
    if editor is not None:
        resolved_block_id = _resolve_block_id("section-header", title, block_id)
        config = editor.section_header(
            resolved_block_id,
            title=title,
            subtitle=subtitle,
            style=style,
        )
        title = str(config["title"])
        subtitle = str(config["subtitle"]) or None
        style = dict(config["style"])
    title_style = _inline_text_style(style, base_size=1.95, color_key="title_color")
    subtitle_style = _inline_text_style(style, base_size=1.0, color_key="body_color")
    if editor is not None:
        title_col, edit_col = st.columns([0.8, 0.2])
        with title_col:
            st.markdown(f'<h3 style="{title_style}">{html.escape(title)}</h3>', unsafe_allow_html=True)
            if subtitle:
                st.markdown(
                    f'<div class="dashboard-section-subtitle" style="{subtitle_style}">{html.escape(subtitle)}</div>',
                    unsafe_allow_html=True,
                )
        with edit_col:
            editor.render_inline_handle(resolved_block_id)
    else:
        st.markdown(f'<h3 style="{title_style}">{html.escape(title)}</h3>', unsafe_allow_html=True)
        if subtitle:
            st.markdown(
                f'<div class="dashboard-section-subtitle" style="{subtitle_style}">{html.escape(subtitle)}</div>',
                unsafe_allow_html=True,
            )


def render_note(title: str, path: Path | None, text: str | None) -> None:
    st.markdown(f"### {title}")
    if path is not None:
        st.caption(f"Source note: `{path}`")
    if text:
        st.markdown(text)
    else:
        st.info("No note file found for this section yet.")


def figure_map(items: list[tuple[str, Path]]) -> dict[str, tuple[str, Path]]:
    return {path.name.lower(): (label, path) for label, path in items}


def get_figure(items: list[tuple[str, Path]], filename: str) -> tuple[str, Path] | None:
    return figure_map(items).get(filename.lower())


def _resolve_figure_path(default_path: Path | None, asset_path: str | None) -> Path | None:
    if asset_path:
        candidate = Path(asset_path)
        if not candidate.is_absolute():
            candidate = repo_root() / candidate
        if candidate.exists():
            return candidate
    return default_path


def _lower_sentence_start(text: str) -> str:
    if len(text) >= 2 and text[0].isupper() and text[1].islower():
        return text[0].lower() + text[1:]
    return text


def _intro_for(text: str, options: list[str]) -> str:
    if not options:
        return ""
    index = sum(ord(char) for char in text) % len(options)
    return options[index]


def _show_sentence(text: str) -> str:
    clean = text.strip()
    lower = clean.lower()
    if lower.startswith(("this view", "this chart", "this figure", "this panel", "here ", "at a glance")):
        return clean
    if lower.startswith("how "):
        return f"{_intro_for(clean, ['Here you can see ', 'This view shows ', 'The chart lays out '])}{clean}"
    return f"{_intro_for(clean, ['This view traces ', 'Here we see ', 'The chart lays out ', 'This panel captures '])}{_lower_sentence_start(clean)}"


def _insight_sentence(text: str) -> str:
    clean = text.strip()
    lower = clean.lower()
    if lower.startswith(("it ", "what stands out", "the main signal", "one clear pattern", "a simple read")):
        return clean
    return f"{_intro_for(clean, ['What stands out is that ', 'A simple read is that ', 'The main signal is that ', 'One clear pattern is that '])}{_lower_sentence_start(clean)}"


def _implication_sentence(text: str) -> str:
    clean = text.strip()
    lower = clean.lower()
    if lower.startswith(("takeaway", "in practice", "for forecasting", "the implication", "this matters")):
        return clean
    return f"{_intro_for(clean, ['In practice, ', 'For forecasting, ', 'The implication is that ', 'This matters because '])}{_lower_sentence_start(clean)}"


def _caption_to_story(caption_content: str | list[tuple[str, str]]) -> str:
    if isinstance(caption_content, str):
        return caption_content.strip()
    parts: list[str] = []
    for lead, text in caption_content:
        if not text:
            continue
        clean = str(text).strip()
        if not clean:
            continue
        lowered = str(lead).strip().lower()
        if "show" in lowered:
            sentence = _show_sentence(clean)
        elif "insight" in lowered or "simple read" in lowered or "read" in lowered:
            sentence = _insight_sentence(clean)
        else:
            sentence = _implication_sentence(clean)
        if sentence and sentence[-1] not in ".!?":
            sentence += "."
        parts.append(sentence)
    return " ".join(parts)


@lru_cache(maxsize=128)
def _static_image_data_uri(path_text: str) -> str:
    path = Path(path_text)
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def render_single_figure(
    item: tuple[str, Path] | None,
    fallback: str,
    caption_content: str | list[tuple[str, str]],
    *,
    editor: PageContentEditor | None = None,
    block_id: str | None = None,
    style: dict[str, object] | None = None,
) -> None:
    if item is None and editor is None:
        st.info(fallback)
        return
    default_label = item[0] if item is not None else fallback.replace("Missing figure: ", "")
    default_path = item[1] if item is not None else None
    default_story = _caption_to_story(caption_content)
    default_asset = ""
    if default_path is not None:
        try:
            default_asset = str(default_path.relative_to(repo_root()))
        except ValueError:
            default_asset = str(default_path)
    if editor is not None:
        resolved_block_id = _resolve_block_id("figure", default_label if default_label else fallback, block_id)
        config = editor.figure_block(
            resolved_block_id,
            label=default_label,
            description=default_story,
            asset_path=default_asset,
            style=style,
        )
        label = str(config["label"])
        story = str(config["description"])
        style = dict(config["style"])
        path = _resolve_figure_path(default_path, str(config["asset_path"]))
        _render_inline_editor_anchor(editor, resolved_block_id)
    else:
        label = default_label
        story = default_story
        path = default_path
    if path is None or not path.exists():
        st.info(fallback)
        return
    label_style = _inline_text_style(style, base_size=1.0, color_key="title_color")
    body_style = _inline_text_style(style, base_size=0.97, color_key="body_color")
    image_uri = _static_image_data_uri(str(path))
    _render_html(
        f"""
        <div class="figure-card-shell">
          <div class="figure-card-label" style="{label_style}">{html.escape(label)}</div>
          <div class="figure-media-shell">
            <img src="{image_uri}" alt="{html.escape(label)}" />
          </div>
          <div class="caption-card"><p style="{body_style}">{html.escape(story)}</p></div>
        </div>
        """
    )


def render_figure_pair(
    items: list[tuple[str, Path]],
    left_filename: str,
    left_caption: str | list[tuple[str, str]],
    right_filename: str,
    right_caption: str | list[tuple[str, str]],
    *,
    editor: PageContentEditor | None = None,
) -> None:
    col1, col2 = st.columns(2, gap="large")
    with col1:
        render_single_figure(
            get_figure(items, left_filename),
            f"Missing figure: {left_filename}",
            left_caption,
            editor=editor,
            block_id=f"figure:{slugify(left_filename)}",
        )
    with col2:
        render_single_figure(
            get_figure(items, right_filename),
            f"Missing figure: {right_filename}",
            right_caption,
            editor=editor,
            block_id=f"figure:{slugify(right_filename)}",
        )


def render_remaining_figures(items: list[tuple[str, Path]], used_filenames: set[str], columns: int = 2) -> None:
    remaining = [(label, path) for label, path in items if path.name.lower() not in used_filenames]
    if not remaining:
        st.info("No appendix figures beyond the curated main-page set.")
        return
    grid = st.columns(columns)
    for index, (label, path) in enumerate(remaining):
        with grid[index % columns]:
            st.image(str(path), caption=label, width="stretch")


def _safe_subset(frame: pd.DataFrame, columns: list[str], rows: int | None = None) -> pd.DataFrame:
    if frame.empty:
        return frame
    keep = [column for column in columns if column in frame.columns]
    subset = frame[keep] if keep else frame
    return subset.head(rows) if rows is not None else subset


def render_table(title: str, frame: pd.DataFrame, columns: list[str] | None = None, rows: int | None = None) -> None:
    if frame.empty:
        return
    st.markdown(f"#### {title}")
    table = _safe_subset(frame, columns or list(frame.columns), rows)
    st.dataframe(table, width="stretch", hide_index=True)


def first_non_null(frame: pd.DataFrame, columns: list[str], default: str = "NA") -> str:
    if frame.empty:
        return default
    row = frame.iloc[0]
    for column in columns:
        if column in frame.columns and pd.notna(row.get(column)):
            value = row.get(column)
            if isinstance(value, (int, float)):
                return format_short_number(value)
            return str(value)
    return default


def best_model_name(comparison: pd.DataFrame, horizon: int) -> str:
    if comparison.empty or "horizon" not in comparison.columns or "model_name" not in comparison.columns:
        return "NA"
    metric = "mean_mase" if "mean_mase" in comparison.columns else None
    if metric is None:
        subset = comparison.loc[comparison["horizon"] == horizon]
        return str(subset.iloc[0]["model_name"]) if not subset.empty else "NA"
    subset = comparison.loc[comparison["horizon"] == horizon].sort_values(metric)
    return str(subset.iloc[0]["model_name"]) if not subset.empty else "NA"


def ranked_models(comparison: pd.DataFrame, horizon: int) -> list[str]:
    if comparison.empty or "horizon" not in comparison.columns or "model_name" not in comparison.columns:
        return []
    subset = comparison.loc[comparison["horizon"] == horizon].copy()
    if subset.empty:
        return []
    metric = "mean_mase" if "mean_mase" in subset.columns else subset.columns[-1]
    subset = subset.sort_values(metric)
    return subset["model_name"].astype(str).tolist()


def candidate_baseline(models: list[str]) -> str:
    preferred = ["seasonal_naive_7", "seasonal_naive", "naive"]
    lowered = {model.lower(): model for model in models}
    for name in preferred:
        if name in lowered:
            return lowered[name]
    return models[1] if len(models) > 1 else (models[0] if models else "NA")


def decision_summary_table(comparison: pd.DataFrame, horizons: list[int], primary: str, secondary: str, directional: str) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for horizon in horizons:
        leader = best_model_name(comparison, horizon)
        if horizon == 7:
            interpretation = primary
        elif horizon == 30:
            interpretation = secondary
        else:
            interpretation = directional
        rows.append({"Horizon": str(horizon), "Leading model": leader, "Interpretation": interpretation})
    return pd.DataFrame(rows)


def model_forecast_table(future: pd.DataFrame, model_name: str, horizon: int, rows: int = 12) -> pd.DataFrame:
    if future.empty:
        return future
    subset = future.loc[
        (future["model_name"].astype(str) == str(model_name))
        & (future["horizon"] == int(horizon))
    ].copy()
    if "date" in subset.columns:
        subset = subset.sort_values("date")
    columns = [
        column
        for column in ["date", "model_name", "horizon", "point_forecast", "prediction", "lower_80", "upper_80"]
        if column in subset.columns
    ]
    return subset[columns].head(rows) if columns else subset.head(rows)


def station_slice_summary(slice_metrics: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if slice_metrics.empty or "horizon" not in slice_metrics.columns:
        return pd.DataFrame()
    subset = slice_metrics.loc[slice_metrics["horizon"] == int(horizon)].copy()
    if subset.empty:
        return subset
    metric = "mean_mase" if "mean_mase" in subset.columns else None
    sort_cols = [column for column in ["slice_type", metric, "model_name"] if column]
    subset = subset.sort_values(sort_cols)
    preferred_cols = [
        "slice_type",
        "slice_value",
        "model_name",
        "mean_mase",
        "mean_mae",
        "mean_rmse",
        "mean_bias",
        "n_series",
        "n_obs",
    ]
    keep = [column for column in preferred_cols if column in subset.columns]
    return subset[keep]


def station_representative_options(future: pd.DataFrame) -> dict[str, list[str]]:
    if future.empty or "station_id" not in future.columns:
        return {}
    options: dict[str, list[str]] = {}
    if "station_category" in future.columns:
        for category in ["busy_stable", "mixed_profile", "sparse_intermittent", "short_history"]:
            station_ids = sorted(
                future.loc[future["station_category"].astype(str) == category, "station_id"].astype(str).unique().tolist()
            )
            if station_ids:
                options[category] = station_ids
    if not options:
        station_ids = sorted(future["station_id"].astype(str).unique().tolist())
        if station_ids:
            options = {"example_1": station_ids[: min(4, len(station_ids))]}
    return options


__all__ = [
    "TOP_HIGHLIGHT_TITLE",
    "best_model_name",
    "bullet_box",
    "candidate_baseline",
    "decision_summary_table",
    "first_non_null",
    "get_figure",
    "metric_cards",
    "model_forecast_table",
    "ranked_models",
    "render_figure_pair",
    "render_note",
    "render_remaining_figures",
    "render_single_figure",
    "render_table",
    "section_header",
    "station_representative_options",
    "station_slice_summary",
    "story_box",
]
