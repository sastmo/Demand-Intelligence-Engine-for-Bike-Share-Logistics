# Station-Level Diagnosis Scripts

Primary runner:

```bash
python3 scripts/station_level/diagnosis/build_station_summary.py \
  --input data/interim/station_level/station_daily.csv \
  --date-col date \
  --station-col station_id \
  --target-col target \
  --n-clusters 6
```

