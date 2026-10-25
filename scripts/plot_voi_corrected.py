"""Raw T2 versus T2C VOI as a fraction of the P0 bill. Numbers come from CSV."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from eci.config import REPO_ROOT


def main() -> int:
    raw_path = REPO_ROOT / "results" / "voi_summary.csv"
    cor_path = REPO_ROOT / "results" / "voi_corrected_summary.csv"
    if not raw_path.exists() or not cor_path.exists():
        print("missing voi summaries", file=sys.stderr)
        return 1
    raw = pl.read_csv(raw_path).filter(pl.col("tariff") == "sce_ev4")
    cor = pl.read_csv(cor_path).filter((pl.col("tariff") == "sce_ev4") & (pl.col("beta") == 0.5))
    penalties = sorted(set(raw["M"].to_list()) | set(cor["M"].to_list()))
    def height(frame: pl.DataFrame, penalty: float) -> float:
        block = frame.filter(pl.col("M") == penalty)
        return float(block["mean_fraction"][0]) if block.height else 0.0

    raw_h = [height(raw, penalty) for penalty in penalties]
    cor_h = [height(cor, penalty) for penalty in penalties]
    positions = np.arange(len(penalties))
    width = 0.36
    figure, axis = plt.subplots(figsize=(6.2, 4.0))
    axis.bar(positions - width / 2, raw_h, width, label="raw T2", color="#8c2d19")
    axis.bar(positions + width / 2, cor_h, width, label="T2C β=0.5", color="#1f4e79")
    axis.set_xticks(positions, [f"M={penalty:g}" for penalty in penalties])
    axis.set_ylabel("VOI / P0 bill")
    axis.set_title("SCE: cost of trusting the deadline")
    axis.legend(frameon=False)
    axis.axhline(0.0, color="0.5", linewidth=0.6)
    figure.tight_layout()
    dest = REPO_ROOT / "plots" / "voi_corrected.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(dest, dpi=140)
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
