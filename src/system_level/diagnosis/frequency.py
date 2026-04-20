from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import periodogram


def summarize_frequency_domain(
    values: pd.Series,
    candidate_periods: tuple[int, ...] = (),
    max_candidates: int = 5,
) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float).to_numpy()
    if len(numeric) < 16:
        return {
            "frequency_domain_status": "insufficient_history",
            "dominant_periods": [],
            "dominant_frequency_peaks": [],
            "matched_candidate_periods": [],
            "primary_spectral_period": None,
            "primary_spectral_peak_power_ratio": None,
            "strong_primary_spectral_peak": False,
        }

    frequencies, power = periodogram(numeric, detrend="linear", scaling="spectrum")
    positive_power = power[frequencies > 0]
    total_positive_power = float(np.sum(positive_power)) if len(positive_power) else 0.0
    median_positive_power = float(np.median(positive_power)) if len(positive_power) else 0.0
    candidates: list[dict[str, float]] = []
    for frequency, magnitude in zip(frequencies, power):
        if frequency <= 0:
            continue
        period = 1.0 / frequency
        if not np.isfinite(period) or period < 2 or period > len(numeric) / 2:
            continue
        matched_candidate = next(
            (
                int(candidate)
                for candidate in candidate_periods
                if abs(period - candidate) <= max(1.0, float(candidate) * 0.15)
            ),
            None,
        )
        candidates.append(
            {
                "period": round(float(period), 2),
                "power": float(magnitude),
                "relative_power": float(magnitude / total_positive_power) if total_positive_power > 0 else 0.0,
                "matched_candidate_period": matched_candidate,
            }
        )

    ranked = sorted(candidates, key=lambda item: item["power"], reverse=True)
    selected: list[dict[str, float]] = []
    for candidate in ranked:
        if all(abs(candidate["period"] - kept["period"]) > max(1.5, kept["period"] * 0.08) for kept in selected):
            selected.append(candidate)
        if len(selected) >= max_candidates:
            break

    matched_candidate_periods = [
        int(item["matched_candidate_period"])
        for item in selected
        if item["matched_candidate_period"] is not None
    ]
    primary_spectral_period = float(selected[0]["period"]) if selected else None
    top_power = float(selected[0]["power"]) if selected else 0.0
    top_power_ratio = float(top_power / median_positive_power) if median_positive_power > 0 and selected else None
    strong_peak = bool(
        selected
        and top_power_ratio is not None
        and top_power_ratio >= 5.0
        and float(selected[0]["relative_power"]) >= 0.10
    )

    return {
        "frequency_domain_status": "ok",
        "dominant_periods": [item["period"] for item in selected],
        "dominant_frequency_peaks": selected,
        "matched_candidate_periods": matched_candidate_periods,
        "primary_spectral_period": primary_spectral_period,
        "primary_spectral_peak_power_ratio": top_power_ratio,
        "strong_primary_spectral_peak": strong_peak,
    }
