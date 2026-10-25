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

## Headline penalty (Day 5 amendment, 2026-10-25T12:15:00Z)

`penalty_unmet: 1000` stays as the stress value. Day 4 showed that number is about 10,000 times the SCE on-peak energy price, so the penalty was 99.7% of the bill and the cost comparison was the unmet term.

`penalty_unmet_headline: 25` USD/kWh is the candidate used for a cost claim. It is about 100 times the SCE summer on-peak price 0.26668 USD/kWh. Twenty-five times that on-peak price is about 6.7 USD/kWh, which is already high for an internal depot; the headline keeps the larger of the two readings so one unmet kWh still dominates the energy price. `results/m_sensitivity.csv` reports M in {10, 25, 100, 1000}. The headline is the largest of those four at which the P3 bill and the P4 bill both have a penalty share under 50%. It is not moved to make P3 beat P1.

A cost claim in the README states this M and the penalty share. A comparison whose penalty share is above 50% is not stated as a cost claim.

## Deadline imputation v2 (Day 5 amendment, 2026-10-25T12:15:00Z)

The Day-4 deadline was connection plus the 2019 median dwell, 5.27 hours. Dwell is right-skewed. On Caltech 2018 sessions with kWh > 0 the median is 4.17 hours, the 95th percentile is 12.36 hours, and the maximum is 245 hours.

T2v2 draws one dwell from the empirical distribution of the same station and Los Angeles hour in that 2018 set. The draw is the quantile of a unit interval hashed from the session id, so the same id always gets the same deadline. A station-hour with fewer than 30 dwells falls back to that hour, then to the whole 2018 distribution. Energy stays `kWhDelivered`. A declared `requestedDeparture` is kept.

## Declared-input bank (Day 5 amendment, 2026-10-25T12:19:18Z)

The 2019 snapshot has no `userInputs`, so a reproduction on `bank_K50` compares two synthetic deadlines. Reproduction v2 uses only Caltech sessions that declared a `requestedDeparture`.

1,195 sessions pass the filters: site 0002, `kWhDelivered > 0`, and a departure delta in (0, 36 hours]. Their local connections run from 2018-04-30 through 2018-11-13. The child bank is a day-block bootstrap of that subset, skeleton year 2018, 50 years, seed 7. `bank_declared_hash` is `1b0a517fadbbc21d25c370454d25f53ec71ae39f05e2eec2de732b540996a70b` (135,566 rows). It was written here before `scripts/reproduce_v2.py` ran. P3 reads the declared departure shifted onto the resampled connection. Energy stays `kWhDelivered`.

## Headline result (Day 5 amendment, 2026-10-25T14:18:16Z)

`results/m_sensitivity.csv` has the four penalties and P0, P1, P2b, P3, P4. No value in that grid puts both P3 and P4 under a 50% penalty share. P3's share is 0.897 at M = 10 and 0.956 at M = 25. P4's share is 0.109 at M = 10 and 0.232 at M = 25. The candidate stays 25. It was not lowered below the grid, and it was not moved to make P3 beat P1. P3 costs more than P0 at every M. Numbers at M = 25 are not cost claims.

P3 unmet under T2v2 is 0.231, above 0.10. The per-scenario diagnosis is `results/t2v2_diagnosis.csv`. Across the 50 years, 52.5% of drawn deadlines are shorter than the true stay, the median draw is 4.49 hours against a median stay of 5.28 hours, and 18.4% of requested kWh does not fit in the drawn window at the plug limit. 0.20% does not fit in the true stay. P0 unmet is 0.187, in line with that short-window shortfall. P1 and P3 unmet are 0.231; the extra energy is planned on slots after the true disconnect and then zeroed by the plant. The draw was not replaced by the median and was not clipped to the disconnect.
