# Results index

Every headline below is a column in the named file. Regenerated numbers live in the CSV, not in this table's prose beyond the pointer.

| file | purpose | headline |
|---|---|---|
| `voi_summary.csv` | Raw declared-deadline VOI vs P0 | `mean_fraction` SCE M=25 = 2.843 (284.3% of P0) |
| `voi_corrected_summary.csv` | T2C β-grid VOI | SCE M=25 β=0.25 `mean_fraction`=1.823; β=0.5=4.886 |
| `voi.csv` / `voi_tests.csv` | Per-scenario VOI and tests | paired with `voi_summary.csv` |
| `voi_ablation.csv` / `voi_ablation_summary.csv` | VOI ablations | demand-off / winter / summer |
| `voi_corrected.csv` / `voi_corrected_tests.csv` / `voi_corrected_finding.csv` | T2C per-scenario + finding row | see summary |
| `metrics_v4.csv` / `metrics_v4_summary.csv` | Frozen-bank T2C policy table | SCE P0 `mean_unmet_rate`=0.0036 |
| `metrics.csv` / `metrics_summary.csv` | Earlier (pre-I9) policy table | historical |
| `reproduction.csv` | v1 gap + min cap | `min_capacity_kw_mean_of_years.repo_value`=79.58 kW; bank-wide max 84.71 kW |
| `reproduction_v2.csv` / `_detail.csv` | Declared-bank plant-still-belief | v2 gap |
| `reproduction_v3.csv` / `_detail.csv` | I9 plant | v3 gap 134.7% |
| `reproduction_v4.csv` / `_detail.csv` / `_m_sweep.csv` | T2C reproduction | `pscc_v4_t2c.repo_value`=2.493 |
| `t2c_dial.csv` | β unmet | β=0.25 / 0.5 / 0.75 unmet % |
| `t2v2_diagnosis.csv` | Pre-I9 P0 unmet | `unmet_p0` mean 18.7% |
| `day6_predictions.csv` | Pre-registered Day-6 bands | `p0_unmet_sce` actual 0.36% |
| `day7_predictions.csv` | Pre-registered Day-7 bands | VOI / v4 gap / dial slope |
| `day6_comparison.csv` | Plant-fix before/after | with `t2v2_diagnosis.csv` |
| `cap_sweep.csv` / `cap_sweep_tests.csv` | Cap × util P1 vs P3 | 60 kW ×4 mean_diff |
| `m_sensitivity.csv` / `_detail.csv` | M grid | 10 / 25 / 100 / 1000 |
| `risk_dial.csv` | Deadline quantile sweep | companion to T2C |
| `dwell_tariff_map.csv` / `_summary.csv` / `_monotonicity_flags.csv` | Dwell × ToU map | scenario analysis |
| `dwell_tariff_break_even.csv` | Break-even dwell | `median_dwell_h`=5.2025 at spread 1 |
| `sobol_indices.csv` / `sobol_detail.csv` | Saltelli ST | `dwell_scale` ST=0.914 |
| `pareto.csv` / `pareto_summary.csv` | ε-peak front | cost vs peak |
| `dlmp_day.csv` / `dlmp_summary.csv` | Capacity duals | `mean_abs_peak_dual_binding`=$0.00577/kW-epoch |
| `forecast_scores.csv` / `forecast_diagnosis.csv` | Rolling-origin pinball | champion LightGBM |
| `hazard_metrics.csv` | Early-unplug / overstay | measured on the mirror |
| `chaos_report.csv` | CH-01 / 06 / 08 | all passed on the synthetic week |
| `e2e_day3.csv` | Day-3 smoke | P0 vs P3 |
| `inference.csv` | Serve-path smoke | latency |
| `flow_benchmark.csv` | Min-cost-flow satellite | vs LP on one day |
| `drift_twente.csv` | PSI/KS monthly windows | `detection_delay_months`=0; `false_triggers`=9; `source`=acn_2018_2020_proxy (Twente NC unreachable) |
| `retrain_demo.csv` | 14-day pinball-ratio trigger | `improvement_pct`=67.76 after refit; `trigger_day`=2020-03-29 |
| `vn_outage_demo.csv` | S1 Bernoulli outage on/off | P0 unmet_kwh 38.84 → 89.02; `outage_hours_served_missed`=11; `i1_ok`=True; params assumed |
