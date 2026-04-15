from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import periodogram


def summarize_frequency_domain(values: pd.Series, max_candidates: int = 5) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float).to_numpy()
    if len(numeric) < 16:
        return {"dominant_periods": [], "dominant_frequency_peaks": []}

    frequencies, power = periodogram(numeric, detrend="linear", scaling="spectrum")
    candidates: list[dict[str, float]] = []
    for frequency, magnitude in zip(frequencies, power):
        if frequency <= 0:
            continue
        period = 1.0 / frequency
        if not np.isfinite(period) or period < 2 or period > len(numeric) / 2:
            continue
        candidates.append({"period": round(float(period), 2), "power": float(magnitude)})

    ranked = sorted(candidates, key=lambda item: item["power"], reverse=True)
    selected: list[dict[str, float]] = []
    for candidate in ranked:
        if all(abs(candidate["period"] - kept["period"]) > max(1.5, kept["period"] * 0.08) for kept in selected):
            selected.append(candidate)
        if len(selected) >= max_candidates:
            break

    return {
        "dominant_periods": [item["period"] for item in selected],
        "dominant_frequency_peaks": selected,
    }
