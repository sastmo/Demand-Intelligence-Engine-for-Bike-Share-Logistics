# System-Level Model Selection

## Context
The system-level daily total is persistent, nonstationary, seasonal, and affected by anomalies. That means the first model stack should be practical, interpretable, and strong on aggregate dynamics before anything more ambitious is added.

## Selected Model Families

| Model | Why selected |
|------|--------------|
| Seasonal Naive | It is the first real benchmark because weekly repeat behavior is visible in the daily total. |
| ETS | Good for stable level, trend, and seasonality structure when the signal is aggregated. |
| SARIMAX / dynamic regression | Good when persistence matters and calendar or known exogenous features should enter directly. |
| Fourier-based dynamic regression | Useful when seasonality is smoother or longer than a single seasonal lag can handle cleanly. |
| Structural state-space / Unobserved Components | Good for separating level, trend, and seasonal structure in a transparent way. |
| LightGBM or XGBoost style lag-feature model | Good as a practical nonlinear challenger when lag, rolling, calendar, and external signals interact. If those libraries are not available, the pipeline falls back to a strong tree-based sklearn model so the system-level workflow still runs end to end. |

## Why These Models Fit This System-Level Problem
- Seasonal Naive keeps the baseline honest.
- ETS is a strong aggregate model when demand has stable recurring structure.
- SARIMAX lets us combine persistence with known drivers like holidays.
- Fourier-based regression helps when longer smooth cycles matter.
- Unobserved Components is useful when the level and trend evolve over time.
- A tree-based lag-feature model is a practical nonlinear challenger without forcing deep learning into the first version.

## Why Some Other Models Are Not First Priority
- Deep models are not the first system-level choice here because this task is only for one aggregated daily series. We want strong interpretable baselines and classical methods first.
- Deep global multi-series models make more sense later when station-level multi-series forecasting is added.
- TBATS was considered because the system-level series likely has multiple seasonalities, but it is deferred for now because implementation stability matters more than adding another dependency-heavy model immediately.
