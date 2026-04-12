---
name: Forecasting Fundamentals Rebuild
about: Rebuild the forecasting work from time-series fundamentals before expanding the app layer
title: "Forecasting Rebuild: Fundamentals-First Time Series Investigation"
labels: ["forecasting", "time-series", "research", "refactor"]
assignees: []
---

## Summary

The current forecasting project has useful structure, but the analytical foundation is still not strong enough.  
Before extending the interactive application further, we need to step back and rebuild the forecasting workflow from first principles:

- understand the time-series behavior deeply
- document cyclic / seasonal / structural-break behavior
- run dedicated diagnostics first
- implement models one by one
- evaluate each model rigorously before promotion

## Why This Work Is Needed

The current project risks moving too quickly into application/UI work before the forecasting engine is fully trustworthy.

Main concerns:

- diagnostics are not yet the clear driver of model choice
- station-level forecasting is not yet mature enough
- interval quality is not yet trustworthy enough
- model comparison and selection need deeper evidence
- the project story should be fundamentals-first, not dashboard-first

## Objective

Create a fundamentals-first forecasting workflow that:

1. investigates the time-series structure first
2. documents the observed seasonal/cyclic behavior
3. identifies structural breaks and regime shifts
4. builds a dedicated diagnostics layer with figures and written findings
5. implements forecasting models one by one
6. evaluates each implementation independently before combining them
7. only then turns the workflow into a stronger interactive application

## Scope

### Phase 1: Diagnostics and Time-Series Characterization

- inspect total and station demand behavior
- quantify trend, seasonality, cyclic behavior, and non-stationarity
- study weekday/weekend differences
- study hourly vs daily vs weekly behavior
- inspect pandemic-era structural breaks
- generate dedicated figures and written notes

### Phase 2: Model-by-Model Implementation

Implement and evaluate each model separately:

- naive baseline
- seasonal naive
- rolling mean baseline
- count GLM
- SARIMAX / Fourier
- weighted ensemble only after model-level evaluation is clear

### Phase 3: Evaluation and Selection

- strict train / validation / test workflow
- rolling-origin backtesting
- interval diagnostics
- regime-aware evaluation
- station-level evaluation

### Phase 4: Application Layer

- update dashboard only after analytical confidence improves
- present diagnostics, model evidence, and forecast outputs more clearly

## Deliverables

- dedicated diagnostics scripts
- dedicated figure outputs
- written diagnostics summary
- one script/module per model family where reasonable
- cleaner model evaluation reports
- clearer justification for why each model is kept, changed, or removed

## Proposed Working Structure

- `src/metro_bike_share_forecasting/diagnostics/`: deeper analysis scripts and reporting
- `src/metro_bike_share_forecasting/forecasting/baselines/`: baseline-by-baseline evaluation
- `src/metro_bike_share_forecasting/forecasting/classical/`: model-by-model classical implementation
- `outputs/figures/`: diagnostics-first figures
- `outputs/reports/`: written conclusions and model comparison outputs

## Acceptance Criteria

- diagnostics are run first and documented clearly
- model choice follows diagnostics rather than guesswork
- each model has its own evidence trail
- station-level behavior is explicitly analyzed
- regime shifts are explicitly analyzed
- evaluation logic is clearly explained
- the interactive app reflects a trustworthy engine rather than leading it

## Notes

This issue is intentionally about rebuilding the analytical foundation first.  
It is not a request for more UI polish by itself.
