"""Grid of synthetic stills. Not camera data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from eci.vlm.fixtures import CLIPS, labels_by_id
from eci.vlm.schema import Detection

COLORS = {
    "lane_split_moto": "#c0392b",
    "oncoming": "#8e44ad",
    "sidewalk_mount": "#d35400",
    "turn_lane_drift": "#2980b9",
    "red_light": "#e74c3c",
    "pedestrian_sudden": "#16a085",
    "none": "#7f8c8d",
}


def render_examples(detections: list[Detection], dest: Path) -> None:
    by_clip = {clip.clip_id: clip for clip in CLIPS}
    labels = labels_by_id()
    chosen = [clip.clip_id for clip in CLIPS[:8]]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for axis, clip_id in zip(axes.ravel(), chosen, strict=True):
        gold = list(labels[clip_id].case_types) or ["none"]
        color = COLORS.get(gold[0], COLORS["none"])
        axis.set_facecolor(color)
        axis.set_xticks([])
        axis.set_yticks([])
        pred = [row.case_type for row in detections if row.clip_id == clip_id]
        caption = by_clip[clip_id].scene_text[:42]
        axis.set_title(f"{clip_id} gold={','.join(gold)}", fontsize=8, color="white")
        axis.text(0.5, 0.55, caption, ha="center", va="center", fontsize=7, color="white")
        axis.text(0.5, 0.12, f"pred={','.join(pred) or 'none'}", ha="center", fontsize=7, color="white")
    fig.suptitle("Self-labeled synthetic urban stills (n=8 of 20). Not nuScenes.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(dest, dpi=120)
    plt.close(fig)
