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

## Zero-kWh sessions (Day 4 amendment)

43.5% of the B1 snapshot has `kWhDelivered == 0`. Those rows occupy a plug and do not request energy.

Decision, recorded before the Day-4 bank run: keep a zero-kWh session when measuring occupancy or congestion, and do not count it as energy need. The frozen bank `bank_hash` `7484043270ba761c133346447998fee64d0bb254b84d30befebcd15365ac1c32` was drawn with `kWh > 0`, so that file has zero-kWh sessions in neither role. The bank is not rebuilt.

## Primary comparison (Day 4 amendment)

The pre-registered primary comparison is P3 against P1 on `sce_ev4`, both at tier T2 (declared inputs plus the imputed deadline). P4 is the T3 oracle and is not in the primary test. The family of secondary comparisons is every other policy pair reported in `results/inference.csv`. Benjamini–Hochberg uses `bh_fdr_q: 0.10`, and the primary test is inside that family so its reported p-value is adjusted.

EVN energy prices are in VND and sit above the frozen numeric penalty if that penalty is read as VND per kWh. The scheduler still delivers every kWh the plugs and the cap allow. Unmet energy, when the cap or the plug limit leaves some, is scored at the frozen `M = 1000` in the tariff's own currency. SCE and EVN totals are not converted into one currency. Both tariffs are applied on the `America/Los_Angeles` clock of the Caltech sessions. The demand charge is added when the bill is scored; it is not a term in the LP.
