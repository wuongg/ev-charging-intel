# ADR 004 — The planning-quantile dial runs backwards

Status: accepted (Day 8). Frozen week-1 numbers live in `results/voi_corrected_summary.csv` and `results/t2c_dial.csv`. This note interprets them. β was not moved after those files were written.

## Decision

Safety in this depot is a **short** planning window, not a long one. The failure mode is the car leaving before the energy is in, not an infrastructure SLA that the operator can stretch. Cheapest-slot will wait for overnight off-peak whenever the believed window is long. If the car has already left, that wait is unmet energy priced at M.

So the planning quantile β of the empirical actual-dwell distribution is a **risk knob**, not a safety knob.

- Low β = assume little time = charge sooner = less unmet.
- High β = assume a long stay = wait for cheap slots = more unmet.

## Evidence (held-out T2C, frozen 2019 bank, SCE)

Trace: `results/voi_corrected_summary.csv`. VOI is cost(P3 at that β) − cost(P4 oracle), as a fraction of the P0 bill.

- Raw declared / T2v2 (Day 6): 284% of P0 at M=25.
- β = 0.25 of actual dwell: 182% at M=25; 84.5% at M=10.
- β = 0.50 (median actual): 489% at M=25.
- β = 0.75: 971% at M=25.

The T2C dial (`results/t2c_dial.csv`) matches: unmet 3.2% / 7.9% / 15.5% at β in {0.25, 0.5, 0.75}. Energy cost falls as β rises; unmet and the bill rise.

Median correction is the wrong point on that dial. It is not that “calibration made things worse.” It is that β=0.5 picks the actual median, which is a long window relative to the early-charge policy that avoids unmet.

## VOI claim that survives the dial

Even the best non-oracle planning assumption in the grid — β=0.25 of the empirical actual-departure distribution — still pays 182% of the P0 bill at M=25 versus an oracle disconnect. That gap is unmet × M (penalty share), not a time-of-use saving. Actual departure-time information is worth more than any single tariff line item. M=10 does not make the gap small.

## What we are not claiming

We are not claiming that β=0.25 is optimal on a finer grid. The grid is {0.25, 0.5, 0.75} and was not tuned. We are not claiming that a short window is “safe” for the driver; it is the assumption that keeps the depot from waiting after the plug is gone.
