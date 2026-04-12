# Fundamentals-First Forecasting Workplan

## Branch

Active branch for this reset:

- `ride/fundamentals-time-series-rebuild`

## Why This Branch Exists

The current project needs a stronger forecasting foundation before we keep expanding the application layer.

This branch is for:

- dedicated time-series diagnostics
- clearer cyclical and seasonal analysis
- model-by-model implementation and review
- stronger written evidence before dashboard expansion

## Recommended Working Order

1. Diagnostics first
2. Write down findings
3. Implement one model at a time
4. Evaluate each model independently
5. Compare models
6. Add ensemble logic after model-level evidence is clear
7. Only then improve the interactive experience

## Suggested Sub-Phases

### Phase A: Time-Series Characterization

- total demand diagnostics
- station demand diagnostics
- intraday/weekday/weekly behavior
- long-cycle / annual behavior
- pandemic / recovery / post-pandemic structure

### Phase B: Baseline Models

- naive
- seasonal naive
- rolling mean

### Phase C: Structured Models

- count GLM
- SARIMAX / Fourier

### Phase D: Selection and Combination

- strict backtest review
- interval diagnostics
- weighted ensemble only if justified

## Useful Git Commands

Create the branch from the current branch:

```bash
git switch -c ride/fundamentals-time-series-rebuild
```

Switch back to it later:

```bash
git switch ride/fundamentals-time-series-rebuild
```

Check current branch and worktree:

```bash
git status --short --branch
```

Push the new branch to origin:

```bash
git push -u origin ride/fundamentals-time-series-rebuild
```

## Suggested First Implementation Tasks

- create dedicated diagnostics-first scripts
- create dedicated figure output folders
- document expected analytical outputs per frequency
- define model-by-model evaluation notes before changing the dashboard

## Important Reminder

Do not treat the app as the source of truth.  
The source of truth should be:

- diagnostics outputs
- model evaluation outputs
- written conclusions
- then the application layer
