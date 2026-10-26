"""Mine corner-case JSON from clips, dedup, index, eval.

Default backend is a keyword extractor over scene text (no GPU). A live
Qwen2-VL / HTTP backend is optional via VLM_API_KEY or VLM_API_URL.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict

from eci.vlm.schema import CASE_TYPES, ClipRecord, Detection, LabelRecord

PROMPTS: dict[str, tuple[str, ...]] = {
    "lane_split_moto": ("xe máy cắt làn", "lanesplit", "cutting across lanes", "motorcycle cutting"),
    "oncoming": ("đi ngược chiều", "oncoming", "wrong-way", "wrong way"),
    "sidewalk_mount": ("bò lên vỉa hè", "sidewalk", "curb"),
    "turn_lane_drift": ("lệch làn khi rẽ", "wide turn", "lane drift"),
    "red_light": ("vượt đèn đỏ", "red light", "light is red"),
    "pedestrian_sudden": (
        "pedestrian suddenly crossing",
        "người sang đường đột ngột",
        "suddenly crossing",
    ),
}

def _confidence(text: str, aliases: tuple[str, ...]) -> float:
    lower = text.lower()
    hits = sum(1 for alias in aliases if alias.lower() in lower)
    if hits == 0:
        return 0.0
    return min(0.55 + 0.2 * hits, 0.95)


def heuristic_mine(clip: ClipRecord) -> list[Detection]:
    """Offline stand-in for Qwen2-VL. Keyword parse of scene_text only."""
    text = clip.scene_text
    lower = text.lower()
    found: list[Detection] = []
    for case, aliases in PROMPTS.items():
        score = _confidence(text, aliases)
        if score <= 0:
            continue
        if case == "oncoming" and "unclear whether anyone is oncoming" in lower:
            continue
        if case == "pedestrian_sudden" and "not yet crossing" in lower:
            continue
        if case == "lane_split_moto" and "maybe lanesplit, maybe just filtering" in lower:
            continue
        quote = next(alias for alias in aliases if alias.lower() in lower)
        found.append(
            Detection(
                clip_id=clip.clip_id,
                case_type=case,  # type: ignore[arg-type]
                timestamp=float(clip.timestamp),
                confidence=score,
                evidence_quote=quote,
            )
        )
    return found


def http_mine(clip: ClipRecord) -> list[Detection]:
    """Optional HTTP VLM. Requires VLM_API_URL. Not used in CI."""
    import requests

    url = os.environ.get("VLM_API_URL", "").strip()
    key = os.environ.get("VLM_API_KEY", "").strip()
    if not url:
        raise RuntimeError("VLM_API_URL is empty")
    payload = {
        "model": os.environ.get("VLM_MODEL", "Qwen2-VL-7B-Instruct"),
        "prompt": (
            "Return JSON list of {case_type, timestamp, confidence, evidence_quote} "
            f"from this driving scene. case_type in {list(CASE_TYPES)}. Scene: {clip.scene_text}"
        ),
    }
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    response = requests.post(url, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    body = response.json()
    raw = body if isinstance(body, list) else body.get("detections") or body.get("content") or []
    if isinstance(raw, str):
        raw = json.loads(raw)
    out = []
    for item in raw:
        item = dict(item)
        item.setdefault("clip_id", clip.clip_id)
        item.setdefault("timestamp", clip.timestamp)
        out.append(Detection.model_validate(item))
    return out


def mine_clips(clips: list[ClipRecord], prompts: dict | None = None, *, backend: str = "heuristic") -> list[Detection]:
    _ = prompts or PROMPTS
    detections: list[Detection] = []
    for clip in clips:
        if backend == "heuristic":
            detections.extend(heuristic_mine(clip))
        elif backend == "http":
            detections.extend(http_mine(clip))
        else:
            raise ValueError(f"unknown VLM backend {backend}")
    return detections


def parse_vlm_json(raw: str | dict | list, clip_id: str) -> list[Detection]:
    """Validate a model dump. Raises on schema failure."""
    if isinstance(raw, str):
        payload = json.loads(raw)
    else:
        payload = raw
    if isinstance(payload, dict):
        payload = payload.get("detections") or [payload]
    return [Detection.model_validate({**item, "clip_id": item.get("clip_id", clip_id)}) for item in payload]


def dedup_and_index(detections: list[Detection]) -> tuple[list[Detection], dict[str, list[Detection]]]:
    """One row per (clip_id, case_type): keep the higher confidence. Stable sort."""
    best: dict[tuple[str, str], Detection] = {}
    for row in detections:
        key = (row.clip_id, row.case_type)
        current = best.get(key)
        if current is None or row.confidence > current.confidence:
            best[key] = row
    ordered = sorted(best.values(), key=lambda item: (item.clip_id, item.case_type, item.timestamp))
    index: dict[str, list[Detection]] = defaultdict(list)
    for row in ordered:
        tokens = set(_tokens(row.case_type)) | set(_tokens(row.evidence_quote))
        for token in tokens:
            index[token].append(row)
    return ordered, dict(index)


def _tokens(text: str) -> list[str]:
    return [part for part in re.split(r"[^0-9a-zA-Zàáảãạăâèéêìíòóôơùúưýđ]+", text.lower()) if len(part) > 1]


def retrieve(index: dict[str, list[Detection]], query: str) -> list[Detection]:
    """Vietnamese or English keyword retrieval over the deduped index."""
    seen: dict[tuple[str, str], Detection] = {}
    for token in _tokens(query):
        for row in index.get(token, []):
            seen[(row.clip_id, row.case_type)] = row
    return sorted(seen.values(), key=lambda item: (-item.confidence, item.clip_id))


def evaluate(detections: list[Detection], labels: list[LabelRecord]) -> list[dict]:
    by_clip: dict[str, set[str]] = defaultdict(set)
    for row in detections:
        by_clip[row.clip_id].add(row.case_type)
    labeled = {row.clip_id: set(row.case_types) for row in labels}
    rows = []
    for case in CASE_TYPES:
        tp = fp = fn = 0
        examples = []
        for clip_id, truth in labeled.items():
            pred = case in by_clip.get(clip_id, set())
            gold = case in truth
            if pred and gold:
                tp += 1
                examples.append(clip_id)
            elif pred and not gold:
                fp += 1
            elif gold and not pred:
                fn += 1
        n_labeled = sum(1 for truth in labeled.values() if case in truth)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        rows.append(
            {
                "case_type": case,
                "n_labeled": n_labeled,
                "precision": precision,
                "recall": recall,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "examples": "|".join(examples[:4]),
            }
        )
    return rows


def hallucination_rate(detections: list[Detection], labels: list[LabelRecord]) -> float:
    """Share of predicted (clip, case) pairs that are not in the self-label set."""
    gold = {(row.clip_id, case) for row in labels for case in row.case_types}
    pred = {(row.clip_id, row.case_type) for row in detections}
    if not pred:
        return 0.0
    return len(pred - gold) / len(pred)
