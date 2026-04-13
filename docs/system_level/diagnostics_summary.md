# System-Level Diagnostics Summary

## Executive Summary
The system-level daily series is forecastable, but it is not simple.

Most important signals:
- strong persistence and autocorrelation
- multiple seasonality, especially weekly plus slower longer-cycle movement
- changing mean over time
- changing variance over time
- real anomalies, spikes, and sudden drops
- calendar and external-driver effects are likely important
- the system-level series is useful for network understanding, but it does not replace future station-level work

## Nature Of The Problem
This is the total daily demand across the whole network. That makes it stable enough for aggregate planning, but it also mixes many behavioral patterns together. The result is a strong series for system-wide forecasting, but not a clean textbook series.

## Main Insights From The Diagnostics
- The series has strong short-memory behavior. Yesterday and last week both matter.
- A seasonal baseline is meaningful here. If a more complex model cannot beat it, that is a warning sign.
- The mean level is not stable. Trend or regime-sensitive logic is needed.
- Variance changes over time, so simple constant-variance assumptions will be fragile.
- The series shows more than one repeating cycle. Weekly structure is clear, and slower monthly or annual movement also appears relevant.
- Spikes and dips are real enough to affect both point forecasts and any later interval work.
- Weekday, weekend, holiday, and broader calendar context are likely important for the total series.

## What We Learned
The system-level problem supports strong baselines, classical statistical models, and lag-based ML models. It does not justify jumping straight to complex deep models. We have enough structure to build a solid first forecasting stack with interpretable methods and a practical ML fallback.

## Risks And Caveats
- A model that assumes one stable mean level will likely underperform.
- A model that ignores calendar structure will miss predictable movement.
- A model that ignores anomalies may look good on average and still fail on operationally important spikes.
- The aggregate series can hide local station behavior, so system-level success should not be confused with network-wide spatial understanding.

## Implications For Modeling
- Start with strong naive and seasonal-naive baselines.
- Use models that can absorb persistence, seasonal structure, and nonstationary behavior.
- Use exogenous calendar and holiday features from the start.
- Add optional external drivers only when they are truly known at forecast time.
- Keep system-level modeling explicit and separate from any future station-level forecasting module.
