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
    figure, axes = plt.subplots(1, len(tariffs), figsize=(4.0 * len(tariffs), 4.0), sharey=False)
    if len(tariffs) == 1:
        axes = [axes]
    for axis, tariff in zip(axes, tariffs, strict=True):
        block = frame.filter(pl.col("tariff") == tariff)
        positions = np.arange(len(penalties))
        heights = [
            float(block.filter(pl.col("M") == penalty)["mean_fraction"][0])
            if block.filter(pl.col("M") == penalty).height
            else 0.0
            for penalty in penalties
        ]
        axis.bar(positions, heights, color=["C0", "C1"][: len(penalties)])
        axis.set_xticks(positions, [f"M={penalty:g}" for penalty in penalties])
        axis.axhline(0.0, color="0.5", linewidth=0.6)
        axis.set_title(str(tariff))
        axis.set_ylabel("VOI / P0 bill")
    figure.suptitle("Value of an oracle deadline")
    figure.tight_layout()
    dest = REPO_ROOT / "plots" / "voi.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(dest, dpi=140)
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
