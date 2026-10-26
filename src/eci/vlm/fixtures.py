"""Twenty self-labeled urban clips. Synthetic frames; not nuScenes redistributed.

Intended source for a later swap: nuScenes mini CAM_FRONT (CC BY-NC-SA).
This fixture set is MIT with the repo: self-drawn stills + self-labeled n=20.
"""

from __future__ import annotations

from pathlib import Path

from eci.vlm.schema import ClipRecord, LabelRecord

# Scene text is what a captioner/VLM would be asked to parse. Some rows are
# deliberately ambiguous so the heuristic backend is not a perfect oracle.
CLIPS: tuple[ClipRecord, ...] = (
    ClipRecord(
        clip_id="c01",
        scene_text="xe máy cắt làn trước mũi xe, lanesplit between two cars at 0.4s",
        timestamp=0.4,
    ),
    ClipRecord(clip_id="c02", scene_text="motorcycle cutting across lanes in dense traffic", timestamp=1.2),
    ClipRecord(clip_id="c03", scene_text="a motorbike weaves; maybe lanesplit, maybe just filtering", timestamp=0.8),
    ClipRecord(clip_id="c04", scene_text="đi ngược chiều trên one-way, oncoming headlight", timestamp=2.0),
    ClipRecord(clip_id="c05", scene_text="vehicle facing us in our lane, wrong-way rider", timestamp=1.1),
    ClipRecord(clip_id="c06", scene_text="bò lên vỉa hè to pass a stall, sidewalk_mount", timestamp=0.6),
    ClipRecord(clip_id="c07", scene_text="scooter climbs the curb onto the sidewalk", timestamp=1.5),
    ClipRecord(clip_id="c08", scene_text="lệch làn khi rẽ phải, wide turn drifts over the line", timestamp=2.2),
    ClipRecord(clip_id="c09", scene_text="car swings wide on a left turn, lane drift", timestamp=0.9),
    ClipRecord(clip_id="c10", scene_text="vượt đèn đỏ at the intersection, red light ignored", timestamp=0.2),
    ClipRecord(clip_id="c11", scene_text="light is red; the lead car still goes through", timestamp=0.3),
    ClipRecord(
        clip_id="c12",
        scene_text="pedestrian suddenly crossing from behind a parked van",
        timestamp=1.8,
    ),
    ClipRecord(clip_id="c13", scene_text="người sang đường đột ngột in front of the bumper", timestamp=0.5),
    ClipRecord(clip_id="c14", scene_text="xe máy cắt làn then vượt đèn đỏ in one motion", timestamp=1.0),
    ClipRecord(clip_id="c15", scene_text="empty arterial, dry asphalt, no road users", timestamp=0.0),
    ClipRecord(clip_id="c16", scene_text="parked cars and a tree shadow, nothing moving", timestamp=0.0),
    ClipRecord(clip_id="c17", scene_text="busy roundabout, unclear whether anyone is oncoming", timestamp=1.4),
    ClipRecord(clip_id="c18", scene_text="child on the curb, not yet crossing", timestamp=0.7),
    ClipRecord(clip_id="c19", scene_text="xe máy cắt làn beside a pedestrian suddenly crossing", timestamp=1.6),
    ClipRecord(clip_id="c20", scene_text="night rain, reflections, no clear violation", timestamp=0.0),
)

LABELS: tuple[LabelRecord, ...] = (
    LabelRecord(clip_id="c01", case_types=["lane_split_moto"], note="clear"),
    LabelRecord(clip_id="c02", case_types=["lane_split_moto"], note="clear"),
    LabelRecord(clip_id="c03", case_types=["lane_split_moto"], note="ambiguous wording"),
    LabelRecord(clip_id="c04", case_types=["oncoming"], note="clear"),
    LabelRecord(clip_id="c05", case_types=["oncoming"], note="clear"),
    LabelRecord(clip_id="c06", case_types=["sidewalk_mount"], note="clear"),
    LabelRecord(clip_id="c07", case_types=["sidewalk_mount"], note="clear"),
    LabelRecord(clip_id="c08", case_types=["turn_lane_drift"], note="clear"),
    LabelRecord(clip_id="c09", case_types=["turn_lane_drift"], note="clear"),
    LabelRecord(clip_id="c10", case_types=["red_light"], note="clear"),
    LabelRecord(clip_id="c11", case_types=["red_light"], note="clear"),
    LabelRecord(clip_id="c12", case_types=["pedestrian_sudden"], note="clear"),
    LabelRecord(clip_id="c13", case_types=["pedestrian_sudden"], note="clear"),
    LabelRecord(clip_id="c14", case_types=["lane_split_moto", "red_light"], note="two events"),
    LabelRecord(clip_id="c15", case_types=[], note="negative"),
    LabelRecord(clip_id="c16", case_types=[], note="negative"),
    LabelRecord(clip_id="c17", case_types=[], note="ambiguous negative"),
    LabelRecord(clip_id="c18", case_types=[], note="near-miss pedestrian, not crossing"),
    LabelRecord(clip_id="c19", case_types=["lane_split_moto", "pedestrian_sudden"], note="two events"),
    LabelRecord(clip_id="c20", case_types=[], note="negative"),
)


def fixture_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


def labeled_clips() -> list[ClipRecord]:
    return list(CLIPS)


def labels_by_id() -> dict[str, LabelRecord]:
    return {row.clip_id: row for row in LABELS}
