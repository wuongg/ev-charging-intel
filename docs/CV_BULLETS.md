# CV bullets — EV Fleet Charging Scheduler & Forecaster

Method-first. No raw percentages. Site names omitted. Every quantity that does appear traces to `results/*.csv` (see [`RESULTS_INDEX.md`](../results/RESULTS_INDEX.md)).

## One line

EV Fleet Charging Scheduler & Forecaster — rolling-horizon LP dispatch (HiGHS) fed by a quantile-forecast ensemble (LightGBM + rearrangement), evaluated under a pre-registered information-tier protocol on 60k+ public charging sessions; quantified the value of departure-time information (VOI) and its sensitivity to declaration bias (ADR-004). FastAPI + Docker; 71 tests; chaos-tested fallbacks.

Trace: session count `docs/DATA_DICTIONARY.md` Measured on this run / sessions; test count is `pytest -q`; ADR-004 `docs/adr/004-risk-knob.md`.

## Two lines

EV Fleet Charging Scheduler & Forecaster — 15-minute MPC with a hard site-cap LP (HiGHS), plant gated on the physical plug (invariant I9), and a quantile forecast ensemble (LightGBM + seasonal naive, embargoed rolling-origin). Evaluated on a frozen K=50 scenario bank drawn from 60k+ public sessions; measured VOI of departure information versus charge-on-arrival and showed that a raw declared deadline is the wrong planning assumption (hazard-corrected T2C, ADR-004). FastAPI + Docker; 71 tests; chaos cases CH-01/06/08.

Trace: bank `results/metrics_v4_summary.csv`; VOI method `results/voi_corrected_summary.csv` (columns `tariff,M,beta,mean_fraction`); plant `results/day6_predictions.csv` metric `p0_unmet_sce`; chaos `results/chaos_report.csv`.

## Three lines

EV Fleet Charging Scheduler & Forecaster — rolling-horizon LP (HiGHS) for workplace dispatch under a site cap, fed by a pinball-scored quantile ensemble (LightGBM + rearrangement) with a 14-day embargo. Information is a tier, not a feature: the plant is the recorded plug (I9); the planner may only see a deadline belief. On a frozen K=50 bank from 60k+ public sessions, VOI of actual departure versus charge-on-arrival is reported as a fraction of the P0 bill (`results/voi_summary.csv` `mean_fraction`; T2C grid in `results/voi_corrected_summary.csv`), and a β risk knob (ADR-004) is the planning assumption, not a post-hoc fit. Add-ons: PSI/KS drift detector with a 14-day retrain trigger (`results/drift_twente.csv`, `results/retrain_demo.csv`); assumed VN unannounced-outage axis (`results/vn_outage_demo.csv`). FastAPI + Docker; 71 tests; chaos-tested fallbacks.
