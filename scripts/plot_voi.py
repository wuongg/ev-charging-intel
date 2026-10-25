"""Grouped bars from results/voi_summary.csv. No numbers are typed in."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from eci.config import REPO_ROOT


def main() -> int:
    path = REPO_ROOT / "results" / "voi_summary.csv"
    if not path.exists():
        print(f"missing {path}", file=sys.stderr)
        return 1
    frame = pl.read_csv(path).sort(["tariff", "M"])
    tariffs = frame["tariff"].unique().sort().to_list()
    penalties = frame["M"].unique().sort().to_list()
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    width = 0.35
    positions = np.arange(len(tariffs))
    for offset, penalty in enumerate(penalties):
        block = frame.filter(pl.col("M") == penalty)
        heights = [
            float(block.filter(pl.col("tariff") == tariff)["mean_fraction"][0])
            if block.filter(pl.col("tariff") == tariff).height
            else 0.0
            for tariff in tariffs
        ]
        axis.bar(positions + (offset - 0.5) * width, heights, width=width, label=f"M={penalty:g}")
    axis.axhline(0.0, color="0.5", linewidth=0.6)
    axis.set_xticks(positions, tariffs)
    axis.set_ylabel("VOI / P0 bill")
    axis.set_title("Value of an oracle deadline")
    axis.legend()
    figure.tight_layout()
    dest = REPO_ROOT / "plots" / "voi.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(dest, dpi=140)
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
