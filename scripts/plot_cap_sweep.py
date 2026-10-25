"""One figure from results/cap_sweep.csv. No numbers are typed in."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import polars as pl

from eci.config import REPO_ROOT


def main() -> int:
    path = REPO_ROOT / "results" / "cap_sweep.csv"
    if not path.exists():
        print(f"missing {path}", file=sys.stderr)
        return 1
    frame = pl.read_csv(path)
    means = (
        frame.group_by(["util", "cap", "policy"])
        .agg(pl.col("cost").mean().alias("cost"), pl.col("p3_vs_p1_gap").mean().alias("gap"))
        .sort(["util", "cap", "policy"])
    )
    utils = sorted(means["util"].unique().to_list())
    figure, axes = plt.subplots(1, len(utils), figsize=(4.2 * len(utils), 4.0), sharey=True)
    if len(utils) == 1:
        axes = [axes]
    for axis, util in zip(axes, utils, strict=True):
        block = means.filter(pl.col("util") == util)
        baseline = block.filter(pl.col("policy") == "P0").select(["cap", "cost"]).rename({"cost": "baseline"})
        for policy, style in (("P1", "-"), ("P3", "-"), ("P4", "o")):
            series = block.filter(pl.col("policy") == policy).join(baseline, on="cap")
            if series.is_empty():
                continue
            reduction = (series["baseline"] - series["cost"]) / series["baseline"]
            axis.plot(series["cap"].to_list(), reduction.to_list(), style, label=policy)
        axis.axhline(0.0, color="0.5", linewidth=0.6)
        axis.set_xlabel("site cap (kW)")
        axis.set_title(f"utilization ×{util}")
        axis.legend()
    axes[0].set_ylabel("cost reduction vs P0")
    figure.suptitle("SCE, T2v2, headline M")
    figure.tight_layout()
    dest = REPO_ROOT / "plots" / "cap_sweep.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(dest, dpi=140)
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
