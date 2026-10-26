# Apply strategy (one pager)

Referral first. Cold apply is the remainder, not the plan. Start outreach **after** tag `v0.9` is on GitHub. `v0.7-week1-frozen` and `v0.8-week2-apply-ready` are already on origin.

Panel split to remember: referral **60–70%** vs cold **30–40%**. Spend the week on people, not on a 12th README adjective.

## Referral channels (concrete)

1. **VinUni / CAIR mentors** — one email: repo URL + two-line EV bullet + which team (GSM platform vs VinFast energy). Ask for a name, not a job.
2. **Vingroup alumni** (LinkedIn school + company filter) — same two-line bullet; ask who owns charging / fleet / data-factory, not “any referral”.
3. **ITviec VinSmart Future JD owner** — in-platform message: 3 sentences (below), attach nothing but the GitHub link.
4. **LinkedIn recruiter** — same 3 sentences; no PDF until they ask.

### 3-sentence outreach

1. I built a workplace EV charging scheduler: 15-minute MPC + site-cap LP, plant vs planner split, pre-registered eval (`github.com/wuongg/ev-charging-intel`, tag `v0.8-week2-apply-ready`).
2. The piece that maps to your JD is [pick one: value-of-information under declaration bias / PSI→retrain SLI / unannounced-outage replay].
3. If useful, I can walk a hiring manager through the frozen bank in 20 minutes — who should see it?

## Team → lead story

| team | lead with | do not lead with |
|---|---|---|
| GSM platform / charging / fleet | EV scheduler (this repo) | license-plate, RAG |
| VinFast energy / factory-ops | EV scheduler + outage/I1 + cap dual | chatbot |
| VinFast ADAS / self-driving | VLM satellite (`docs/README_vlm.md`) + RAG | charging VOI |
| VinDynamics MLOps / data factory | drift PSI + 14-day pinball retrain + freeze tags | ToU savings |
| Speech / LLM product | Traffic-law multi-agent RAG | this repo |

One lead story per team. The second project is the keyword match (RAG for LLM JDs, VLM for ADAS).

## CV variant matrix

Use v5 + `docs/CV_V6_NOTES.md`. Only the **summary clause** and **project order** change.

| variant | summary clause | project order |
|---|---|---|
| GSM-platform | decision systems under constraints for fleet charging and ops | EV → RAG → (plate optional, one line) |
| VinFast-ops-energy | decision systems under constraints (MPC, site cap, outage replay) | EV → RAG; omit plate |
| VinDynamics-MLOps | decision systems under constraints with drift SLIs and freeze gates | EV (drift/retrain sentence) → RAG; VLM satellite only if they ask data-factory / video |

## Timeline

- **Now:** `v0.7` and `v0.8` are on GitHub. Do not wait to *prepare* names; do wait to *send* until `v0.9` if the satellite eval CSV is in the email pack.
- **After `v0.9`:** start the four channels that week. Two referral asks per day beats twenty cold clicks.
- **Do not** rewrite `v0.7` / `v0.8`. New work is a new tag.
