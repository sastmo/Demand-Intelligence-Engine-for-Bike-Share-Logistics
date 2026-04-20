## Scope Summary

| Item | Definition |
|---|---|
| Scope | Aggregated network-wide daily demand series only |
| Target | Total daily demand across all stations |
| Aggregation Level | System-level total only |
| Frequency | Daily |
| Station-Level Forecasting | Not included in this module |

## Forecast Horizons

| Horizon | Purpose |
|---|---|
| 7 days | Main operational planning horizon |
| 30 days | Main tactical planning horizon |
| 90 days | Extended sensitivity horizon |

## Constraints

- Forecasts should remain nonnegative.
- Only features known at prediction time should be used.
- Evaluation must stay time-aware.

## Decision Use Case

- Aggregate planning
- System-wide resource planning
- Network-level monitoring
- Strategic demand tracking

## External Drivers

External drivers are optional, but the contract is designed to accept:

- holidays
- weather / temperature
- promotions or campaigns
- event flags

If a future external driver is not known at forecast time, it should be excluded or handled with an explicit assumption.

## Validation Design

| Rule | Requirement |
|---|---|
| Backtesting approach | Rolling-origin / walk-forward only |
| Random split | Not allowed |
| Evaluation structure | Horizon-based evaluation for 7-day and 30-day forecasts |

## Metrics

| Category | Metrics |
|---|---|
| Primary | MAE, RMSE, MASE |
| Secondary | Bias / mean error, coverage, mean interval width if intervals are added later |

MAPE is not the primary metric for this contract.

## Model Families In Scope

- Naive baseline
- Seasonal naive baseline
- ETS
- SARIMAX / dynamic regression
- Fourier-based dynamic regression
- Structural state-space / Unobserved Components
- Tree-based ML with lagged, rolling, calendar, holiday, and optional external features

## Out of Scope For Now

- Station-level forecasting
- Deep multi-series forecasting
- Fleet allocation or routing optimization
- Real-time online learning
- Regime-specific production logic beyond the current aggregate baseline setup
