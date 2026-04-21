# System-Level Demand Diagnosis

![EDA](https://img.shields.io/badge/EDA-Data%20Analysis-2F80C9?style=flat-square)
![Time Series](https://img.shields.io/badge/Time%20Series-Diagnosis-2F80C9?style=flat-square)
![System Level](https://img.shields.io/badge/System%20Level-Forecasting-69BE45?style=flat-square)
![Frequency Domain](https://img.shields.io/badge/Frequency%20Domain-Seasonality-E88A45?style=flat-square)
![Multi-Horizon](https://img.shields.io/badge/Multi--Horizon-7d%20%2F%2030d%20%2F%2090d-2F80C9?style=flat-square)
![Uncertainty](https://img.shields.io/badge/Uncertainty-Intervals-2F80C9?style=flat-square)

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

<table>
  <tr>
    <td width="50%" valign="top">
      <h3>3. Autocorrelation</h3>
      <img src="outputs/figures/acf.png" alt="Autocorrelation plot" width="100%">
      <p><em>Correlation of the series with its past lags.</em></p>
    </td>
    <td width="50%" valign="top">
      <h3>4. Seasonal Pattern</h3>
      <img src="outputs/figures/seasonal_profile.png" alt="Day-of-week or seasonal pattern" width="100%">
      <p><em>Repeating demand pattern across the weekly cycle.</em></p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h3>5. Frequency-Domain View</h3>
      <img src="outputs/figures/periodogram.png" alt="Periodogram or frequency spectrum" width="100%">
      <p><em>Weekly, monthly, and yearly seasonal structure in the signal.</em></p>
    </td>
    <td width="50%" valign="top">
      <h3>6. Decomposition</h3>
      <img src="outputs/figures/stl.png" alt="Trend, seasonal, residual decomposition" width="100%">
      <p><em>Trend, seasonal, and residual components of the series.</em></p>
    </td>
  </tr>
</table>

## Main Takeaways
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
