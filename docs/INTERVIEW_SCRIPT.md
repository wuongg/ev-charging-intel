# 90-second script + cold answers

Numbers are copied from `results/*.csv`. If a CSV changes, rewrite this file from the CSV, not from memory.

## 90 seconds

I built a workplace EV charging scheduler that is honest about information.

The plant is the plug. The planner is a 15-minute rolling-horizon LP (HiGHS) with a hard site cap. Those are different objects. On the frozen bank, treating a believed deadline as the unplug left 18.7% of energy unmet (`results/t2v2_diagnosis.csv`, column `unmet_p0`). After the plant used the recorded disconnect, P0 unmet is 0.36% (`results/day6_predictions.csv`, metric `p0_unmet_sce`, column `actual`). That bug is the story I would tell about real telematics: the first thing that breaks is writing a user-declared window into the meter.

Value of information is not a ToU trick. On SCE at M=25, a raw declared deadline costs 284.3% of charge-on-arrival (`results/voi_summary.csv`, `tariff=sce_ev4`, `M=25`, `mean_fraction=2.843`). The best non-oracle planner, β=0.25 of empirical dwell, still costs 182.3% (`results/voi_corrected_summary.csv`, `tariff=sce_ev4`, `M=25`, `beta=0.25`, `mean_fraction=1.823`). Median dwell as the belief (β=0.5) is worse: 488.6% (`beta=0.5`, `mean_fraction=4.886`). Safety is assuming little time. ADR-004.

Capacity: 79.6 kW mean year, 84.7 kW serves every bank day (`results/reproduction.csv`, metrics `min_capacity_kw_mean_of_years` and `min_capacity_kw_serves_every_bank_day`, column `repo_value`). Scheduling intelligence on energy cost pays when median dwell is at least 5.2 h (`results/dwell_tariff_break_even.csv`, `median_dwell_h=5.2025`). Sobol total-order on dwell_scale is 0.91 (`results/sobol_indices.csv`, `factor=dwell_scale`, `ST=0.914`). Binding-hour peak dual ≈ $0.006/kW-epoch (`results/dlmp_summary.csv`, `value=0.00577`).

The same PSI/KS detector that flags a regime shift (COVID proxy, delay 0 months, `results/drift_twente.csv` `detection_delay_months`) is the retrain trigger: pinball ratio > 1 for 14 days, then refit, 67.8% pinball cut (`results/retrain_demo.csv`, `improvement_pct`, `trigger_day=2020-03-29`). That is the drift pattern a JD for monitoring actually asks for.

## Cold answers

### G.1 — What is the plant versus the policy?

The plant is `[connection, disconnect)` and `kWhDelivered`. Invariant I9. The policy may believe a shorter or longer window. If I let the belief gate the meter, charge-on-arrival stops early and I invent a 18.7% unmet rate that is not physics (`t2v2_diagnosis.csv` `unmet_p0` vs `day6_predictions.csv` `p0_unmet_sce`).

### G.2 — Why LP, not MILP?

Continuous power, linear unmet, hard cap. HiGHS. Integer sockets are a different problem (the flow satellite). ADR-001. On this bank, 150 kW does not bind, so P3 matches P1 (`metrics_v4_summary.csv`). The LP earns its keep when the cap binds (`cap_sweep_tests.csv`) or when a peak dual is the price (`dlmp_summary.csv`).

### G.3 — How do you stop leakage?

Session-fold T2C: a session is scored with the fold that was not fit on it. Forecast embargo 336 h; lags use actuals strictly before the hour. Frozen banks hashed in `results/FREEZE.md`. If an LLM writes a deadline into the plant, that is a methodology failure, not a wording issue.

### G.4 — What is VOI here?

VOI is (cost of a deadline belief − cost of charge-on-arrival) as a fraction of the P0 bill, penalty included. It is unmet × M, not ToU savings. Trace `voi_summary.csv` / `voi_corrected_summary.csv` `mean_fraction`. β is a risk knob, not a fit target (ADR-004).

### G.5 — What do you do when the forecast or the fleet is wrong?

Chaos: CH-01 drop 10% sessions, CH-06 infeasible cap → priced unmet, CH-08 ignore-pilot then recover (`chaos_report.csv`). Drift: PSI on daily kWh and hourly counts, KS on residuals, frozen quantile bins, ε=1e-4 (`drift_twente.csv`). Retrain when pinball ratio vs own baseline exceeds 1 for 14 days (`retrain_demo.csv`).

### What breaks first with real V-Green data?

The plant-fix story, not the LP. Telematics will give a declared or predicted departure that is not the unplug. If that timestamp is wired into the meter, P0 unmet jumps from 0.36% to 18.7% and every penalty number is fiction. Wire the plug. Treat the declaration as a belief. Then the drift detector is how you notice the next distribution shift — COVID on the public proxy fired the same month (`drift_twente.csv` `detection_delay_months=0`, `retrain_demo.csv` `trigger_day=2020-03-29`).
