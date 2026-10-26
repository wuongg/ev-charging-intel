# VLM scenario-mining satellite

Add-on after `v0.7-week1-frozen`. Core charging results are unchanged.

This is a **data engine sketch for ADAS validation** (VinFast Self-Driving / data-factory JD) and a real-to-sim mining loop (VinDynamics / Skild-style video learning). It is not a perception paper.

## Setup

| knob | this machine |
|---|---|
| GPU | none required for the committed demo |
| Weights | Qwen2-VL-7B-Instruct is the intended open model (Hugging Face) |
| Backend used for `results/vlm_mining_eval.csv` | `heuristic` — keyword parse of scene text (offline stand-in) |
| Live VLM | set `VLM_API_URL` + `VLM_API_KEY` and `VLM_BACKEND=http` |

No torch extra is in the default lock. CI skips the HTTP test unless `VLM_API_KEY` is present (`skip-with-reason`).

## Data

Intended source: **nuScenes mini** CAM_FRONT (CC BY-NC-SA; do not redistribute). The mini split is too large for this repo and for CI.

Committed demo: **20 synthetic urban stills**, **self-labeled, n=20**. Captions live in `src/eci/vlm/fixtures.py`. This is not a public ADAS benchmark.

Taxonomy (VN-relevant): xe máy cắt làn / lanesplit, đi ngược chiều, bò lên vỉa hè, lệch làn khi rẽ, vượt đèn đỏ, pedestrian suddenly crossing.

## Run

```bash
uv run python scripts/run_vlm_mining.py
```

Writes `results/vlm_mining_eval.csv` and `plots/vlm_examples.png`.

## Eval (self-labeled n=20)

Trace: `results/vlm_mining_eval.csv`. Backend `heuristic`. Hallucination rate **5.9%** (`hallucination_rate`) — predicted (clip, case) pairs not in the label set (here: `sidewalk_mount` on an ambiguous curb frame).

| case_type | n_labeled | precision | recall |
|---|---:|---:|---:|
| lane_split_moto | 5 | 1.00 | 0.80 |
| oncoming | 2 | 1.00 | 1.00 |
| sidewalk_mount | 2 | 0.67 | 1.00 |
| turn_lane_drift | 2 | 1.00 | 1.00 |
| red_light | 3 | 1.00 | 1.00 |
| pedestrian_sudden | 3 | 1.00 | 1.00 |

Limitation: self-labeled n=20; heuristic backend is not Qwen2-VL; VLM hallucination rate on ambiguous frames = 5.9% in this set. Do not cite as a model leaderboard.

## Swap

Point `ClipRecord.path` at nuScenes JPEG frames and set `VLM_BACKEND=http` against a Qwen2-VL-7B-Instruct endpoint. Schema (`Detection`) and `dedup_and_index` stay the freeze-gate for the satellite.
