"""Heatmap of P3 reduction vs P0. Numbers come from the summary CSV."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from eci.config import REPO_ROOT


def main() -> int:
    path = REPO_ROOT / "results" / "dwell_tariff_map_summary.csv"
    if not path.exists():
        print(f"missing {path}", file=sys.stderr)
        return 1
    frame = pl.read_csv(path)
    scales = sorted(frame["dwell_scale"].unique().to_list())
    spreads = sorted(frame["tariff_spread"].unique().to_list())
    grid = np.zeros((len(spreads), len(scales)))
    for i, spread in enumerate(spreads):
        for j, scale in enumerate(scales):
            cell = frame.filter((pl.col("tariff_spread") == spread) & (pl.col("dwell_scale") == scale))
            grid[i, j] = float(cell["mean_energy_reduction_vs_p0"][0]) * 100.0 if cell.height else 0.0
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    image = axis.imshow(grid, origin="lower", cmap="RdBu", aspect="auto")
    axis.set_xticks(range(len(scales)), [str(v) for v in scales])
    axis.set_yticks(range(len(spreads)), [str(v) for v in spreads])
    axis.set_xlabel("dwell scale")
    axis.set_ylabel("tariff spread")
    for i in range(len(spreads)):
        for j in range(len(scales)):
            axis.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", color="black", fontsize=8)
    figure.colorbar(image, ax=axis, label="P3 energy reduction vs P0 (%)")
    axis.set_title("Dwell × tariff spread, energy only (scenario analysis)")
    figure.tight_layout()
    dest = REPO_ROOT / "plots" / "dwell_tariff_map.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(dest, dpi=140)
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
