# Current Pipeline Health

This note is about the current system-level forecasting path only.

## Current State

- the system-level pipeline runs end to end
- the repo now separates diagnosis from forecasting more clearly
- ETS is still a healthy short-horizon benchmark
- Fourier dynamic regression is a strong medium-horizon model in the current setup
- SARIMAX has improved, but it is still a candidate that needs careful monitoring rather than blind trust
- 90-day outputs should be treated as sensitivity review, not as the main validated production horizon

## What Is Healthy

- clear `system_level` naming across configs, docs, code, and outputs
- rolling backtests for the main validated horizons
- probabilistic intervals built from out-of-sample residual calibration
- explicit system-level output area under `forecasts/system_level/`

## Known Weak Spots

| Issue | Evidence | Risk | Action |
|------|----------|------|--------|
| SARIMAX still needs stabilization | medium-horizon performance is not consistently better than ETS or Fourier | unstable production behavior if over-promoted | keep it as a monitored candidate |
| 30-day quality is usable but not yet strong | current results are workable, not best-in-class | medium-horizon planning may still drift | focus next model work on model quality, not UI |
| 90-day is not production-grade | only treated as sensitivity | users may overread long-horizon numbers | keep the wording conservative |
| branch structure was previously inconsistent | diagnosis and forecasting paths were mixed | confusion during iteration | keep using the standardized layout now in place |

## Practical Guidance

- use system-level diagnosis under `diagnosis/system_level_analysis/`
- use system-level forecasting under `src/metro_bike_share_forecasting/system_level/` and `scripts/system_level/forecasting/`
- read forecast outputs only from `forecasts/system_level/`
- do not mix station-level diagnosis outputs with system-level forecast outputs
