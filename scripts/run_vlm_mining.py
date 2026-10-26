"""Run the VLM satellite on the 20 self-labeled synthetic clips."""

from __future__ import annotations

import csv
import os
import sys

from eci.config import REPO_ROOT, provenance
from eci.vlm.fixtures import CLIPS, LABELS
from eci.vlm.mining import dedup_and_index, evaluate, hallucination_rate, mine_clips
from eci.vlm.plot import render_examples


def main() -> int:
    backend = os.environ.get("VLM_BACKEND", "heuristic")
    detections = mine_clips(list(CLIPS), backend=backend)
    ordered, _index = dedup_and_index(detections)
    rows = evaluate(ordered, list(LABELS))
    hallu = hallucination_rate(ordered, list(LABELS))
    prov = provenance()
    dest = REPO_ROOT / "results" / "vlm_mining_eval.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_type",
        "n_labeled",
        "precision",
        "recall",
        "tp",
        "fp",
        "fn",
        "examples",
        "backend",
        "n_clips",
        "hallucination_rate",
        "label_note",
        *prov.keys(),
    ]
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload.update(prov)
            payload["backend"] = backend
            payload["n_clips"] = len(CLIPS)
            payload["hallucination_rate"] = hallu
            payload["label_note"] = "self-labeled n=20 synthetic stills; not a public ADAS benchmark"
            writer.writerow(payload)
    plot = REPO_ROOT / "plots" / "vlm_examples.png"
    render_examples(ordered, plot)
    print(f"wrote {dest} backend={backend} hallucination={hallu:.3f}")
    print(f"wrote {plot}")
    for row in rows:
        print(
            f"{row['case_type']} n={row['n_labeled']} "
            f"P={row['precision']:.2f} R={row['recall']:.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
