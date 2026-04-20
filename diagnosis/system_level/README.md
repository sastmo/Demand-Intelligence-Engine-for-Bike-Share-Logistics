# System-Level Demand Diagnosis

<div style="display:flex;flex-wrap:wrap;gap:6px 8px;margin:6px 0 14px 0;">
  <span style="display:inline-flex;border-radius:6px;overflow:hidden;font-family:Arial,sans-serif;font-size:11px;line-height:1;">
    <span style="background:#555555;color:#FFFFFF;padding:5px 9px;">EDA</span>
    <span style="background:#2F80C9;color:#FFFFFF;padding:5px 9px;">Data Analysis</span>
  </span>
  <span style="display:inline-flex;border-radius:6px;overflow:hidden;font-family:Arial,sans-serif;font-size:11px;line-height:1;">
    <span style="background:#555555;color:#FFFFFF;padding:5px 9px;">Time Series</span>
    <span style="background:#2F80C9;color:#FFFFFF;padding:5px 9px;">Diagnosis</span>
  </span>
  <span style="display:inline-flex;border-radius:6px;overflow:hidden;font-family:Arial,sans-serif;font-size:11px;line-height:1;">
    <span style="background:#555555;color:#FFFFFF;padding:5px 9px;">System Level</span>
    <span style="background:#69BE45;color:#FFFFFF;padding:5px 9px;">Forecasting</span>
  </span>
  <span style="display:inline-flex;border-radius:6px;overflow:hidden;font-family:Arial,sans-serif;font-size:11px;line-height:1;">
    <span style="background:#555555;color:#FFFFFF;padding:5px 9px;">Frequency Domain</span>
    <span style="background:#E88A45;color:#FFFFFF;padding:5px 9px;">Seasonality</span>
  </span>
  <span style="display:inline-flex;border-radius:6px;overflow:hidden;font-family:Arial,sans-serif;font-size:11px;line-height:1;">
    <span style="background:#555555;color:#FFFFFF;padding:5px 9px;">Multi-Horizon</span>
    <span style="background:#2F80C9;color:#FFFFFF;padding:5px 9px;">7d / 30d / 90d</span>
  </span>
  <span style="display:inline-flex;border-radius:6px;overflow:hidden;font-family:Arial,sans-serif;font-size:11px;line-height:1;">
    <span style="background:#555555;color:#FFFFFF;padding:5px 9px;">Uncertainty</span>
    <span style="background:#2F80C9;color:#FFFFFF;padding:5px 9px;">Intervals</span>
  </span>
</div>

A short diagnostic summary of aggregate bike-sharing demand and what it means for forecasting.

## What This Covers
This view treats the full network as one time series. At the system level, the signal is smoother than station-level demand and easier to read for trend, seasonality, and forecasting direction.

## Key Signal Views

### 1. Aggregate Demand Over Time
![Figure 1 - Aggregate demand over time](outputs/figures/series.png)

_Overall system demand across the full observation window._

### 2. Demand Distribution
![Figure 2 - Demand distribution](outputs/figures/distribution.png)

_Distribution of aggregate demand values._

### 3. Autocorrelation
![Figure 3 - Autocorrelation plot](outputs/figures/acf.png)

_Correlation of the series with its past lags._

### 4. Seasonal Pattern
![Figure 4 - Day-of-week or seasonal pattern](outputs/figures/seasonal_profile.png)

_Repeating demand pattern across the weekly cycle._

### 5. Frequency-Domain View
![Figure 5 - Periodogram or frequency spectrum](outputs/figures/periodogram.png)

_Weekly, monthly, and yearly seasonal structure in the signal._

### 6. Decomposition
![Figure 6 - Trend, seasonal, residual decomposition](outputs/figures/stl.png)

_Trend, seasonal, and residual components of the series._

## Main Readout
- The aggregate demand signal is forecastable and not random.
- The series shows persistence and clear seasonality.
- Frequency analysis suggests recurring weekly, monthly, and yearly structure.
- Forecast quality changes by horizon, so model claims should be horizon-specific.
- Multi-horizon forecasting is needed because operational, planning, and directional decisions happen on different time windows.

## Forecasting Direction

| Model / Layer | Best Use | Strength | Watchout |
|---|---|---|---|
| Seasonal Naive | Baseline for short repeating cycles | Simple and transparent | Limited flexibility |
| ETS / Classical Models | Strong short-horizon benchmark | Stable and interpretable | Can miss changing patterns |
| Fourier / Regression / SARIMAX | Medium-horizon candidate | Captures smoother seasonal structure | Needs monitoring |
| Probabilistic Layer | All horizons | Adds forecast intervals | Depends on point forecast quality |

## Why Multi-Horizon Forecasting
A 7-day forecast supports near-term operations. A 30-day forecast supports planning. A 90-day forecast is more useful as a directional view and should be interpreted more cautiously.

## Why Not Only Point Forecasts
Point forecasts are useful, but they do not show the likely range around the prediction. The next step should keep prediction intervals alongside the main forecast.

## Next Step
Use this diagnosis as the foundation for model benchmarking, horizon-specific evaluation, and interval design.
