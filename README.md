# EV Fleet Charging Intelligence

Workplace charging, from the plug event to a feasible schedule: mirror the sessions, audit them, then forecast and dispatch.

Day 1 delivers the mirror, the audit gate, and CI. The forecast tables and the first scheduler number are below. Simulation beyond the K=50 bank, and the API, are still stubs.

## Day 1

```bash
make install
make data      # Kaggle mirror (no token) + audit gate
make test
```

`ACN_DATA_TOKEN` switches the download to the Caltech API. Without it, the pipeline uses the public Kaggle mirror and recovers sessions from the truncated JSON instead of crashing.

Provenance on every generated artifact: `git_sha`, `config_hash`, `data_hash`, `utc_now`.

SCE and EVN prices in `configs/` are filed rates with source links in the YAML comments. SCE is TOU-EV-4 as of March 2019 (`sce_tou_ev_4_march_2019` in ACN-Sim). EVN energy uses the October 2025 trial bands and the QĐ 963 clock.

## Forecast (Day 2, v1)

Rolling-origin scores on `mirror_2018` (11 origins, 3696 hours). On that mirror, `primary_2019_06_2020_03` and `regime_2020_03` have 0 origins; the mirror ends 2018-11-13. Chronos was not fit. These rows are the information-asymmetric run: lag features inside the training window were null across the horizon.

| model | mean pinball | coverage | p-value vs naive |
|---|---:|---:|---:|
| seasonal-naive | 6.148 | 0.780 | baseline |
| LightGBM | 7.501 | 0.388 | 1.00 |
| ensemble | 5.778 | 0.738 | 0.0005 |

Equal-weight ensemble has the lowest mean pinball and beats seasonal-naive (p=0.0005); LightGBM alone is worse than seasonal-naive (p=1) and its q10-q90 coverage is 0.388, outside [0.70, 0.90].

## Forecast (Day 3, v2)

Same protocol on the B1+B2 series (`extended_2018_2020`, 65 origins, 21840 hours, 2018-04-25 through 2020-12-02). Embargo is still 336 hours and still restricts training rows only. Lag and rolling features at hour t use actuals before t. Chronos was not fit. The pre-registered primary and regime windows now contain 19 origins each; those counts are in `results/forecast_scores.csv` and were not fit a second time.

| model | mean pinball | coverage | p-value vs naive |
|---|---:|---:|---:|
| seasonal-naive | 3.779 | 0.823 | baseline |
| LightGBM | 2.407 | 0.753 | 0.0005 |
| ensemble | 2.891 | 0.835 | 0.0005 |

LightGBM has the lowest mean pinball and beats seasonal-naive (p=0.0005, improvement 36.3%). Its overall q10-q90 coverage is 0.753, inside [0.70, 0.90]. The equal-weight ensemble also beats seasonal-naive and loses to LightGBM. 24 of 65 origins have coverage outside [0.70, 0.90]; per-origin coverage and a null-feature fraction of 0 are in `results/forecast_diagnosis.csv`.

## Scheduler (Day 3)

P3 (LP-MPC) reaches cost 115.25 versus P0 128.23 on the day-1 scenario (−10.1%). Those dollars are the pre-verification rate card. Trace: `results/e2e_day3.csv`.

## Reproduction

Same frozen bank, filed SCE TOU-EV-4. The PSCC gap is (cost of P3 at T2 − cost of P4 at T3) / cost of P4. Minimum capacity is the smallest site cap that delivers every kWh the plugs can deliver. Trace: `results/reproduction.csv`.

| metric | literature | repo | delta |
|---|---:|---:|---:|
| PSCC gap, P3 T2 vs P4 T3 | 9.2% | 2918.4% | +2909.2 pp |
| minimum site cap, mean of the 50 years | 60 kW | 79.58 kW | +19.58 kW |

The repo gap is the ratio 29.184 stored in that CSV, which is 2918.4%. It was not tuned toward 9.2%. The bank year has no declared `userInputs`, so every T2 deadline is connection plus the 2019 Caltech median dwell, 5.27 hours, and unmet energy is priced at the frozen M = 1000 USD/kWh. Mean unmet is 11.0% of requested energy for P3 and 0.36% for P4.

Three ablations are in the same file. Removing the demand charge moves the ratio from 29.18 to 31.06, so the demand charge is not the source. Keeping only sessions whose imputed deadline falls inside the true stay moves it to 55.35, so early departures are not the source. P3 run with T3 information is the P4 schedule, and that gap is 0. The excess over 9.2% is the imputed deadline. The cap that serves every bank day, not the mean year, is 84.71 kW (+24.71 kW against 60).

## Results

Mean cost reduction against P0 across the 50 scenarios. A negative value means the policy costs more than charge-on-arrival. Intervals are the 95% CI of that mean. Trace: `results/metrics_summary.csv`, tests in `results/inference.csv`.

| tariff | policy | mean reduction vs P0 | 95% CI |
|---|---|---:|---|
| sce_ev4 | P1 | −209.9% | [−213.5%, −206.2%] |
| sce_ev4 | P2b | 0.0% | [0, 0] |
| sce_ev4 | P3 | −209.9% | [−213.5%, −206.2%] |
| sce_ev4 | P4 (T3 oracle) | +89.0% | [+88.3%, +89.8%] |
| evn_2026 | P1 | +1.61% | [+1.57%, +1.65%] |
| evn_2026 | P2b | 0.0% | [0, 0] |
| evn_2026 | P3 | +1.61% | [+1.57%, +1.65%] |
| evn_2026 | P4 (T3 oracle) | +1.49% | [+1.37%, +1.61%] |

The pre-registered primary comparison is P3 against P1 on SCE, inside the Benjamini–Hochberg family at q = 0.10. The two policies have the same cost on every scenario (mean difference 0, p = 1, adjusted p = 1, not rejected). Mean site peak is 118.7 kW and the plug-binding fraction is 0, against the 150 kW cap, so the cap does not bind and the LP commits the same cheapest-slot plan as P1. On EVN the secondary comparison of P3 against P0 is a 1.61% reduction (adjusted p = 1.5×10⁻⁵²). P3 and P1 match there as well.

CH-01, CH-06, and CH-08 pass in CI. Trace: `results/chaos_report.csv`.
