"""Structured VLM output. Invalid JSON is a pipeline failure, not a soft miss."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

CASE_TYPES = (
    "lane_split_moto",
    "oncoming",
    "sidewalk_mount",
    "turn_lane_drift",
    "red_light",
    "pedestrian_sudden",
)

CaseType = Literal[
    "lane_split_moto",
    "oncoming",
    "sidewalk_mount",
    "turn_lane_drift",
    "red_light",
    "pedestrian_sudden",
]


class Detection(BaseModel):
    clip_id: str
    case_type: CaseType
    timestamp: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_quote: str

    @field_validator("evidence_quote")
    @classmethod
    def quote_not_blank(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("evidence_quote is empty")
        return text


class ClipRecord(BaseModel):
    clip_id: str
    path: str = ""
    scene_text: str
    timestamp: float = 0.0


class LabelRecord(BaseModel):
    clip_id: str
    case_types: list[CaseType]
    note: str = ""
