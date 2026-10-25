# EV Fleet Charging Intelligence

Workplace charging, from the plug event to a feasible schedule: mirror the sessions, audit them, then forecast and dispatch.

Day 1 delivers the mirror, the audit gate, and CI. The forecast table is below. Scheduler, simulation, and the API are still stubs that exit 0.

## Day 1

```bash
make install
make data      # Kaggle mirror (no token) + audit gate
make test
```

`ACN_DATA_TOKEN` switches the download to the Caltech API. Without it, the pipeline uses the public Kaggle mirror and recovers sessions from the truncated JSON instead of crashing.

Provenance on every generated artifact: `git_sha`, `config_hash`, `data_hash`, `utc_now`.

Tariff prices in `configs/` are placeholders. Do not quote a cost until those TODOs are replaced with the tariff sheet.

## Forecast (Day 2)

Rolling-origin scores on `mirror_2018` (11 origins, 3696 hours). `primary_2019_06_2020_03` and `regime_2020_03` have 0 origins; the mirror ends 2018-11-13. Chronos was not fit.

| model | mean pinball | coverage | p-value vs naive |
|---|---:|---:|---:|
| seasonal-naive | 6.148 | 0.780 | baseline |
| LightGBM | 7.501 | 0.388 | 1.00 |
| ensemble | 5.778 | 0.738 | 0.0005 |

Equal-weight ensemble has the lowest mean pinball and beats seasonal-naive (p=0.0005); LightGBM alone is worse than seasonal-naive (p=1) and its q10-q90 coverage is 0.388, outside [0.70, 0.90].
