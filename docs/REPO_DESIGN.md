# EV Fleet Charging Intelligence — Repository Design Doc (v1.0)

> Trước khi code: cấu trúc repo, quy trình, tên file/hàm/module, data contracts, test plan.
> Nguồn thiết kế: architecture_final_v3/v4/v5 (20 agent, 4 vòng phản biện).
> Timeline: 4 tuần + satellites. Quy tắc vàng: **results-first, freeze-gate, 1-command regeneration.**

---

## 1. Repo tree (toàn bộ, có chú thích)

```
ev-charging-intel/
├── README.md                    # 90-second surface: 1 sentence + 1 figure + 1 results table + 3 numbers
├── LICENSE                      # MIT
├── Makefile                     # make data / forecast / optimize / evaluate / chaos / serve / test
├── pyproject.toml               # project metadata + deps (uv/pip)
├── uv.lock                      # pinned lockfile — committed DAY 1
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml               # pytest + ruff + nightly mini-pipeline (200 sessions)
│
├── configs/                     # ZERO magic numbers ở đây — mọi thứ tham số hóa
│   ├── site_caltech.yaml        # site params: EVSE count=54, DCFC 50kW, site cap 150kW, tz
│   ├── site_superstation.yaml   # 1,038-port VN scenario (S5)
│   ├── tariff_sce_ev4.yaml      # SCE TOU EV-4 (data-native)
│   ├── tariff_evn_2026.yaml     # QĐ 963: peak 17:30–22:30 T2–T7, off-peak 0:00–6:00
│   ├── tariff_evn_twopart.yaml  # regime D: capacity 209,459–286,153 đ/kW/mo + energy
│   ├── forecast.yaml            # model params, quantiles [0.1,0.5,0.9], lags, embargo
│   ├── scheduler.yaml           # slot 15min, penalty M, epsilon_peak, committed_k
│   ├── scenarios.yaml           # scenario bank spec: K=50, strata, utilization grid {1,2,4,8}
│   │                            #   dwell grid, congestion arms, VN axes S1–S5
│   ├── chaos.yaml               # CH-01..CH-10 fault injections (scenario DSL)
│   └── protocol.yaml            # PRE-REGISTRATION: primary comparison, metrics, seeds, BH-FDR q=0.10
│
├── data/                        # gitignored; manifest chỉ lưu HASH + ngày tải
│   ├── manifest.yaml            # sha256 + source + date cho mỗi snapshot
│   ├── raw/                     # acn_sessions_YYYYMMDD.json.gz, twente/, weather/, paloalto/
│   └── interim/                 # sessions.parquet (cleaned + flag columns)
│
├── src/eci/                     # package name: eci = ev-charging-intel
│   ├── __init__.py
│   ├── config.py                # load_config() → pydantic Settings; schema validation; config_hash()
│   ├── constants.py             # slot length, columns, flags, invariants enum
│   │
│   ├── data/                    # ============ L0: DATA ============
│   │   ├── __init__.py
│   │   ├── ingest.py            # fetch_acn_site(site, token) → raw json.gz; paginate _links.next
│   │   ├── download.py          # download_twente(), download_weather(), download_paloalto()
│   │   ├── schema_contract.py   # coerce types (siteID→str), assert dtypes, golden-schema
│   │   ├── parse.py             # parse_sessions() → tz-aware UTC + local (zoneinfo); RFC1123 %Z
│   │   ├── clean.py             # clean_sessions() → funnel R1..V8 + flag columns
│   │   ├── audit.py             # audit_gate() → audit_report.html + quality_score Q=Π(1−rate)
│   │   ├── dictionary.py        # build_data_dictionary() → DATA_DICTIONARY.md table rows
│   │   ├── measures.py          # measure_hazards() → early-unplug gap, ICED rate, overstay rate,
│   │   │                        #   multi-input rate, overdispersion test (var/mean hourly)
│   │   └── features.py          # build_hourly_demand() → hourly kWh attributed by CONNECTION time
│   │                            #   + right-censoring rule; build_userinputs_snapshot()
│   │
│   ├── forecast/                # ============ L1: FORECAST (nhỏ có chủ ý) ============
│   │   ├── __init__.py
│   │   ├── naive.py             # seasonal_naive(series, m=168) → df + quantiles via residual hist
│   │   ├── lightgbm_q.py        # train_lightgbm_quantile() (alpha=τ, pinball); SHARED hyperparams
│   │   ├── chronos.py           # chronos_zero_shot() → q10/q50/q90 (timebox 2h, CPU fallback)
│   │   ├── ensemble.py          # equal_weight_quantiles() + rearrange() [np.sort] +
│   │   │                        #   inverse_pinball_weights() (pre-registered challenger)
│   │   ├── backtest.py          # rolling_origin_cv() → folds, embargo=max session duration,
│   │   │                        #   inner-fold Optuna; leakage-audit hook (shuffled-time control)
│   │   ├── metrics.py           # pinball_loss(), empirical_coverage(), mae(), paired_block_test()
│   │   │                        #   (moving-block bootstrap B=2000, block=168h; DM-HAC secondary)
│   │   └── select.py            # pick_forecast() → champion/challenger logic + demote rule
│   │
│   ├── optimize/                # ============ L2: SCHEDULER (trái tim) ============
│   │   ├── __init__.py
│   │   ├── state.py             # FleetState: plugged EVs, remaining energy (STATEFUL carryover),
│   │   │                        #   willingness windows (≠ occupancy), running_peak (METERED floor)
│   │   ├── formulation.py       # build_lp() → time-indexed LP: p[i,t], unmet[i], P_peak epigraph;
│   │   │                        #   constraints: q90 derated cap, SLA 90%, C-rate, SoC-dwell ~90%
│   │   ├── solve.py             # solve_epoch() → scipy.optimize.milp wrapper; returns rates+status+
│   │   │                        #   duals (ineqlin.marginals = shadow prices/DLMP)
│   │   ├── warmstart.py         # HighsSession: getBasis/setBasis qua highspy; churn metric
│   │   ├── policies.py          # P0 charge_on_arrival() · P1 cheapest_slots_greedy() (cost-aware)
│   │   │                        #   P2b edf()/llf() (ACN-Sim suite, feasibility axis)
│   │   ├── flow.py              # build_flow_network() + solve_mincost_flow() (OR-Tools)
│   │   │                        #   [satellite — SAU freeze gate]
│   │   ├── feasibility.py       # check_feasibility(schedule, state) — INDEPENDENT checker:
│   │   │                        #   re-verifies cap/deadline/energy từ đầu; gates EVERY emission
│   │   ├── mpc.py               # MPCLoop: 15-min epochs, re-solve từ ACTUALS, committed first-k
│   │   │                        #   slots lock, info tier T1(clairvoyant)/T2(declared+impute)/T3(oracle)
│   │   └── objective.py         # cost terms: Σπ_t·p + c_d·P_peak + M·Σunmet; ε-peak tie-breaker
│   │
│   ├── simulate/                # ============ L3: SIMULATION (pyramid) ============
│   │   ├── __init__.py
│   │   ├── bootstrap.py         # day_block_bootstrap() — whole-session TRIPLES, stratified
│   │   │                        #   weekday×quarter, primary regime 2019-01→2020-02
│   │   ├── ipp.py               # fit_nhpp() (λ_m(t) per stratum) — second engine; thinning sampler
│   │   ├── scenario_bank.py     # make_scenario_bank(K=50) → save disk + SHA BEFORE any policy run
│   │   ├── congestion.py        # CongestionProtocol 3 arms: queue_edf (PRIMARY) / balk /
│   │   │                        #   mixed(q_max); plug=54 binding resource; balk/renege events
│   │   ├── des_lite.py          # event-ordered plug occupancy trong slot engine
│   │   │                        #   (Unplug-before-Plugin, ACN-Sim semantics)
│   │   ├── queueing.py          # erlang_b_balk(), erlang_a_renege() — PSA closed-form (L1)
│   │   ├── hazard.py            # apply_hazards() — early-unplug axis, ICED, overstay,
│   │   │                        #   misdeclaration (lognormal μ1.1 σ0.3 + strategic arm)
│   │   ├── vn_scenarios.py      # S1 outage Bernoulli (season rate, unannounced 1–6h)
│   │   │                        #   S2 tariff-regime switch mid-run · S3 monsoon duty (rain +22-25%)
│   │   │                        #   S4 heat-wave coupling · S5 superstation scale (1,038 ports)
│   │   ├── variance.py          # control_variates() (P0 daily kWh/session count); CRN harness
│   │   └── engine.py            # run_scenario() — 15-min slot loop, glue L2 policies + faults
│   │
│   ├── eval/                    # ============ L4: EVALUATION HARNESS ============
│   │   ├── __init__.py
│   │   ├── metrics.py           # cost_reduction(), optimality_gap() (P4↔P3), unmet_rate(),
│   │   │                        #   peak_kw(), par(), jain_index(), churn(), solve_time_p95()
│   │   ├── sobol.py             # sobol_indices() (SALib; LHS design) trên util×dwell×tariff
│   │   ├── pareto.py            # epsilon_sweep() → cost-vs-peak front, per q50/q90
│   │   ├── reproduce.py         # reproduce_pscc_9p2() · reproduce_min_capacity_60kw()
│   │   │                        #   → Δ-audit table vs published
│   │   ├── drift.py             # psi_ks_detector() (inputs + residuals, frozen quantile bins,
│   │   │                        #   ε-floor); retrain_trigger(); Twente regime-shift demo
│   │   ├── chaos.py             # run_chaos_suite() → fault injectors per interface (data/forecast/
│   │   │                        #   tariff/plant/solver); assert invariants I1–I8 EVERY run
│   │   ├── invariants.py        # INVARIANTS I1–I8 — importable, dùng bởi cả checker lẫn chaos
│   │   └── stats.py             # paired_ci() (over seeds + day-clustered), bh_fdr() (q=0.10)
│   │
│   ├── serve/                   # ============ L5: SERVING (mỏng) ============
│   │   ├── __init__.py
│   │   ├── app.py               # FastAPI: POST /optimize · GET /forecast · GET /health
│   │   ├── schemas.py           # pydantic request/response (typed)
│   │   └── mcp_tools.py         # [v1.5, SAU freeze gate] wrap endpoints thành MCP tools
│   │
│   └── viz/
│       ├── __init__.py
│       └── plots.py             # static matplotlib: dwell×tariff 2D map, Pareto front,
│                                #   shadow-price curve, PSA-vs-simulated balk overlay
│
├── scripts/                     # CLI entrypoints (mỗi Makefile target gọi 1 script)
│   ├── download_data.py         # → data/raw + manifest hashes (day 1 MANDATORY)
│   ├── run_audit.py             # → audit_report + sessions.parquet (gate)
│   ├── train_forecasts.py       # → models/ + forecast_scores.csv
│   ├── build_scenarios.py       # → scenarios/bank_K50.parquet + hash
│   ├── run_policies.py          # → results/raw/…parquet (per policy per cell)
│   ├── evaluate.py              # → results/metrics.csv + git_sha + config_hash + data_hash
│   ├── run_chaos.py             # → results/chaos_report.csv
│   ├── reproduce.py             # → results/reproduction.csv (Δ vs PSCC/ACN published)
│   └── make_report.py           # → README figures + docs/report.md (full thesis)
│
├── models/                      # gitignored; lightgbm .txt artifacts + forecast_scores.csv
├── scenarios/                   # gitignored; bank_K50.parquet + scenario_bank_hash
├── results/                     # COMMITTED: metrics.csv, chaos_report.csv, reproduction.csv
│
├── tests/                       # pytest — signal cao nhất/h
│   ├── test_feasibility.py      # INDEPENDENT MILP/LP feasibility checker (test #1)
│   ├── test_invariants.py       # I1–I8 từng cái 1
│   ├── test_schema.py           # golden schema (siteID str, dtype, tz parse)
│   ├── test_tariff.py           # golden-day regression: frozen day optimal cost bất biến qua PR
│   ├── test_forecast_leakage.py # shuffled-time control ≈ naive; snapshot invariant
│   ├── test_flow_equivalence.py # flow vs HiGHS gap ≤ 1e-6 [satellite]
│   ├── test_chaos_smoke.py      # CH-01/06/08 mini versions trong CI
│   └── conftest.py              # fixtures: tiny synthetic sessions, frozen tariff
│
└── docs/
    ├── DATA_DICTIONARY.md       # mọi cột, unit, tz, known issues (5.23% clock-skew…)
    ├── PROTOCOL.md              # pre-registration 15 mục — freeze TRƯỚC final runs
    ├── report.md                # full thesis (README chỉ là 90-second surface)
    └── adr/                     # architecture decision records ngắn
        ├── 001-lp-not-milp.md   #   "LP deliberately: bang-bang is optimal"
        ├── 002-mincost-flow.md  #   TU proof (satellite)
        └── 003-soft-sla.md      #   M·unmet ⇒ feasible by construction
```

## 2. Data flow & naming conventions

**Data flow một chiều, không vòng:**
```
raw/ → ingest → parse → clean(+flags) → [AUDIT GATE] → interim/sessions.parquet
     → features → forecast models → scenario bank → MPC replay → results/*.csv → README figures
```

**Quy ước:**
- Module = tầng kiến trúc (data/forecast/optimize/simulate/eval/serve). Hàm = động từ + danh từ: `clean_sessions`, `solve_epoch`, `check_feasibility`.
- **Mỗi file output sinh ra phải kèm provenance**: `git_sha`, `config_hash`, `data_hash` — hàm `provenance()` trong `config.py` trả dict này, mọi script gắn vào results row đầu.
- Flags trong Parquet tiền tố `f_`: `f_clock_skew`, `f_long_dwell`, `f_ui_absurd`, `f_dst_cross`, `f_type_coerced`.
- Policies: hàm nhận `(state: FleetState, config) → Schedule`; không hàm nào đọc disk trực tiếp.
- **Config là nguồn sự thật duy nhất** — code không chứa số; tariff sống trong YAML versioned + effective_date.

## 3. Quy trình làm việc (workflow per week)

- **Day 1 (hard checklist):** token ACN requested 9:00 → `scripts/download_data.py` mirror EOD (nếu token trễ: dùng Kaggle mirror dextersebs/ev-dataset đã verify + ghi truncation) → `uv.lock` commit → CI skeleton → **ugly E2E run by day 3** (1 tariff, 1 site, P0 vs P3, 1 số).
- **Mỗi tuần kết thúc = 1 bảng kết quả regenerate từ 1 lệnh** (`make evaluate`) — nếu không, tuần chưa xong.
- **Freeze gate trước satellites:** `README v1 + results/metrics.csv frozen` HOẶC không đụng flow.py/mcp_tools.py/VLM.
- **Kill list (P0>P1>P2>P3):** P0 = reproduction + headline cost · P1 = dwell map + Pareto · P2 = Sobol, warm-start, drift · P3 = satellites. Mất tuần → cắt từ dưới lên, không thảo luận.
- **Git:** trunk-based; branch `feat/<module>`; PR = CI green + feasibility test pass; tag `v0.1-week1`… mỗi tuần.
- **Pre-registration:** `configs/protocol.yaml` + `docs/PROTOCOL.md` freeze TRƯỚC final runs (sau khi pipeline debug xong, trước khi chạy K=50).

## 4. Test plan chi tiết (đã map từ 4 vòng phản biện)

| Test | Kiểm chứng gì | Nguồn phát hiện |
|---|---|---|
| `test_feasibility` | Independent checker re-verify cap/deadline/energy post-solver | MLOps round-1 (#1 signal) |
| `test_invariants` | I1–I8 từng cái | Robustness round-4 |
| `test_schema` | siteID str, dtype, tz parse, arrival peak 7–10am | Data-audit round-4 |
| `test_tariff` | Golden-day cost bất biến qua refactor; schema windows non-overlap | Panel Q9 |
| `test_forecast_leakage` | Shuffled-time control ≈ naive; snapshot invariant trên features | Methodology H7 + pre-mortem S4 |
| `test_flow_equivalence` | flow vs HiGHS ≤1e-6 | OR round-3 |
| `test_chaos_smoke` | CH-01/06/08 mini | Robustness round-4 |

## 5. API contracts chính (signatures)

```python
# config.py
def load_config(path: Path) -> Settings
def config_hash(cfg: Settings) -> str

# data/ingest.py
def fetch_acn_site(site: str, token: str, start: str, end: str) -> Iterator[list[dict]]

# data/clean.py
def clean_sessions(raw: pl.DataFrame) -> pl.DataFrame   # + f_* flags + funnel row

# data/features.py
def build_hourly_demand(sessions: pl.DataFrame, site: str) -> pl.DataFrame  # conn-time attribution

# forecast/ensemble.py
def equal_weight_quantiles(members: dict[str, pl.DataFrame]) -> pl.DataFrame  # + rearrange

# optimize/state.py
class FleetState:  # energy_carried, willingness windows, running_peak (metered floor)

# optimize/mpc.py
class MPCLoop:
    def run(self, scenario: Scenario, policy: Policy, tier: Literal["T1","T2","T3"]) -> ReplayLog

# simulate/engine.py
def run_scenario(scenario: Scenario, policy: Policy, faults: list[Fault] = []) -> RunResult

# eval/metrics.py
def optimality_gap(p3_cost: float, p4_cost: float) -> float
def jain_index(delivered: np.ndarray, required: np.ndarray) -> float
```

## 6. Makefile targets

```makefile
make data        # download + audit gate (block nếu fail)
make forecast    # train 3 members + ensemble + backtest table
make scenarios   # scenario bank K=50 + hash
make optimize    # run_policies: P0/P1/P2b/P3(×3 tiers)/P4 × cells
make evaluate    # metrics.csv + provenance + BH-FDR  ← 1-command regeneration
make chaos       # chaos_report.csv (CH-01..10; 3 cái trong CI)
make reproduce   # reproduction.csv (PSCC 9.2% + 60kW)
make plots       # static figures → README
make serve       # docker compose up (FastAPI)
make test        # pytest -q
```

## 7. Trình tự code (dependency order để luôn có thể chạy E2E)

1. `config.py` + `constants.py` + tariff YAMLs + golden tariff test
2. `data/ingest→parse→clean→audit` + `test_schema` (day 1–2)
3. `data/features.py` (hourly, conn-time) — đủ để forecast chạy
4. `forecast/naive→lightgbm_q→backtest` (day 3–4) → **bảng forecast đầu tiên**
5. `optimize/formulation+solve+feasibility` + `test_feasibility` (day 5–6) → **P0 vs P3 số đầu**
6. `simulate/bootstrap+engine` (day 7–8) → E2E ugly run hoàn chỉnh
7. `optimize/policies` (P1, P2b) + `mpc.py` (tiers) (tuần 2)
8. `eval/reproduce` — PSCC +9.2% + 60kW (tuần 2, gate cho mọi claim)
9. `simulate/congestion+hazard+vn_scenarios` (tuần 2–3)
10. `eval/pareto+sobol+stats` (tuần 3)
11. `eval/chaos+invariants` (tuần 2–3, CH-01/06/08 sớm)
12. `serve/app` + `viz/plots` + README 90-second surface (tuần 4)
13. Sau freeze gate: `optimize/flow` → `serve/mcp_tools` → VLM satellite (tuần 5+)
```
