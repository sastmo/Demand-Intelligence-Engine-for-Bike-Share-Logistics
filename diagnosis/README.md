# Diagnosis

This folder contains diagnosis-only work.

It is split by scope:

- `system_level_analysis/`
  system-level time-series diagnosis
- `station_level_analysis/`
  station summary, categorization, clustering, and diagnosis reporting

Rule:

- diagnosis artifacts stay inside the relevant diagnosis folder
- forecasting artifacts do not belong here
- live implementation code should stay in `src/`
- runnable entrypoints should stay in `scripts/`

Current output locations:

- `diagnosis/system_level_analysis/outputs/`
- `diagnosis/station_level_analysis/outputs/`
