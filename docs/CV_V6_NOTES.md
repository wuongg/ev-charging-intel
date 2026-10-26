# CV v6 notes — edit the v5 file you already own

Do not invent new metrics. Copy wording from `docs/CV_BULLETS.md` and skills already shown in the README. You keep the Word/PDF; this file is only the delta.

## 1. Project order

Keep **Vietnam Traffic Law Chatbot (Multi-Agent RAG)** in the list, but insert **above it**:

**EV Fleet Charging Scheduler & Forecaster** — use the **two-line** bullet from `docs/CV_BULLETS.md` as-is:

> EV Fleet Charging Scheduler & Forecaster — 15-minute MPC with a hard site-cap LP (HiGHS), plant gated on the physical plug (invariant I9), and a quantile forecast ensemble (LightGBM + seasonal naive, embargoed rolling-origin). Evaluated on a frozen K=50 scenario bank drawn from 60k+ public charging sessions; measured VOI of departure information versus charge-on-arrival and showed that a raw declared deadline is the wrong planning assumption (hazard-corrected T2C, ADR-004). FastAPI + Docker; 75 tests; chaos cases CH-01/06/08.

Repo line (optional, one URL): `https://github.com/wuongg/ev-charging-intel` (tags `v0.7-week1-frozen`, `v0.8-week2-apply-ready`).

## 2. Skills rows to add

**Optimization & Simulation:** Rolling-horizon LP (HiGHS), min-cost-flow (OR-Tools), MPC, simulation with CRN, Sobol sensitivity, Pareto analysis.

**Model Risk & Eval:** PSI/KS drift detection, pre-registered protocols, paired block bootstrap, chaos testing.

Do not add unpublished VLM leaderboard numbers. If a VinFast ADAS variant needs a line, point to `docs/README_vlm.md` as a satellite, not as a core result.

## 3. Summary

Add the phrase **decision systems under constraints** (one competency). Example splice, not a new claim:

`… building decision systems under constraints: information-tier charging dispatch, RAG assistants, and eval harnesses.`

## 4. Ordering by destination

| slot | keep | drop / shrink |
|---|---|---|
| 1 | EV Fleet Charging Scheduler & Forecaster | — |
| 2 | Vietnam Traffic Law Chatbot (Multi-Agent RAG) | — |
| 3 | License-plate / vision demo | **Public GSM / VinFast ops-energy:** one line or omit (saturated). Keep for other applies if you want. |

Lead story by team is in `docs/APPLY_STRATEGY.md`. Do not lead with this charging repo for ADAS/speech.

## 5. What not to add

- No new percentages that are not in `results/*.csv`.
- No “Caltech” / site names.
- No VLM precision as a headline skill until a real Qwen2-VL run replaces the heuristic backend.
