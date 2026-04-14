# Station-Level Analysis

This folder is for station-level diagnosis artifacts and notes.

It does not contain the live implementation code anymore.

Use this runner:

```bash
python3 scripts/station_level/diagnosis/build_station_summary.py \
  --input data/interim/station_level/station_daily.csv \
  --date-col date \
  --station-col station_id \
  --target-col target \
  --n-clusters 6
```

Outputs are written here:

- `diagnosis/station_level_analysis/outputs/tables/`
- `diagnosis/station_level_analysis/outputs/diagnostics/`
- `diagnosis/station_level_analysis/outputs/reports/`
