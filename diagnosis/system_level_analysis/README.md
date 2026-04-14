# System-Level Analysis

This folder is for diagnosis only.

It is the home for:

- system-level time-series diagnostics
- diagnostic runner scripts
- temporary figures, tables, and markdown reports used before modeling decisions

Main entrypoint:

```bash
python3 scripts/system_level/diagnosis/run_diagnostics.py --synthetic-demo --target-col value --frequency daily
```

Output location:

- `diagnosis/system_level_analysis/outputs/figures/`
- `diagnosis/system_level_analysis/outputs/tables/`
- `diagnosis/system_level_analysis/outputs/reports/`

This folder does not own forecasting outputs.
System-level forecast artifacts belong in `forecasts/system_level/`.
