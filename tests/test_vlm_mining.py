"""Satellite tests. Live VLM HTTP is skipped unless VLM_API_KEY is set."""

from __future__ import annotations

import json
import os

import pytest

from eci.vlm.fixtures import CLIPS, LABELS
from eci.vlm.mining import dedup_and_index, evaluate, mine_clips, parse_vlm_json
from eci.vlm.schema import Detection


def test_schema_rejects_blank_quote() -> None:
    with pytest.raises(Exception):
        Detection(
            clip_id="c01",
            case_type="red_light",
            timestamp=0.1,
            confidence=0.9,
            evidence_quote="  ",
        )


def test_parse_vlm_json_list() -> None:
    raw = json.dumps(
        [
            {
                "case_type": "oncoming",
                "timestamp": 1.0,
                "confidence": 0.8,
                "evidence_quote": "đi ngược chiều",
            }
        ]
    )
    rows = parse_vlm_json(raw, "c04")
    assert len(rows) == 1
    assert rows[0].clip_id == "c04"
    assert rows[0].case_type == "oncoming"


def test_dedup_is_deterministic() -> None:
    messy = [
        Detection(
            clip_id="c01",
            case_type="lane_split_moto",
            timestamp=0.1,
            confidence=0.4,
            evidence_quote="first",
        ),
        Detection(
            clip_id="c01",
            case_type="lane_split_moto",
            timestamp=0.2,
            confidence=0.9,
            evidence_quote="second",
        ),
        Detection(
            clip_id="c01",
            case_type="lane_split_moto",
            timestamp=0.3,
            confidence=0.5,
            evidence_quote="third",
        ),
    ]
    first, _ = dedup_and_index(messy)
    second, _ = dedup_and_index(list(reversed(messy)))
    assert [row.model_dump() for row in first] == [row.model_dump() for row in second]
    assert len(first) == 1
    assert first[0].confidence == 0.9
    assert first[0].evidence_quote == "second"


def test_heuristic_eval_has_precision_and_recall() -> None:
    detections = mine_clips(list(CLIPS), backend="heuristic")
    ordered, index = dedup_and_index(detections)
    rows = evaluate(ordered, list(LABELS))
    assert len(rows) == 6
    assert all("precision" in row and "recall" in row for row in rows)
    assert any(row["n_labeled"] > 0 and row["recall"] > 0 for row in rows)
    assert any(index.values())


@pytest.mark.skipif(not os.environ.get("VLM_API_KEY"), reason="VLM_API_KEY not set")
def test_live_vlm_api_optional() -> None:
    detections = mine_clips(list(CLIPS)[:1], backend="http")
    for row in detections:
        Detection.model_validate(row.model_dump())
