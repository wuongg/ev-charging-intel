# Protocol

Pre-registration skeleton. `configs/protocol.yaml` is the machine-readable copy.

## Regime window (Day 3 amendment)

The Day-2 Kaggle mirror ended 2018-11-13, so the pre-registered primary window (origins 2019-06-01 through 2020-03-01) and the COVID regime (origins from 2020-03-01) had 0 origins. Path B1, the ACN-Data-Static snapshot of California_Garage_01 and Arroyo_Garage_01, extends the session span through 2020-12-03.

The scenario bank uses calendar 2019, the first full pre-COVID year inside that span. `regime.start = 2019-01-01`, `regime.end = 2020-01-01` (exclusive). Days are resampled whole, stratified by weekday and quarter.

## Common random numbers

`bank_hash` in `configs/protocol.yaml` is the sha256 of `scenarios/bank_K50.parquet`. Every policy run reads that file. The hash is also an entry in `data/manifest.yaml`.

## Frozen on Day 3

These were not tuned to the end-to-end number:

- embargo stays 336 hours and restricts training rows only
- unmet penalty `M = 1000` USD/kWh in `configs/scheduler.yaml`
- committed horizon `k = 1` slot
