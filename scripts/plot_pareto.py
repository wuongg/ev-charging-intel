"""Cost vs peak front and a one-day cap dual strip. Numbers come from CSV."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import polars as pl

from eci.config import REPO_ROOT


def plot_front() -> int:
    path = REPO_ROOT / "results" / "pareto.csv"
    if not path.exists():
        print(f"missing {path}", file=sys.stderr)
        return 1
    frame = pl.read_csv(path)
    figure, axis = plt.subplots(figsize=(6.2, 4.2))
    for util in sorted(frame["util"].unique().to_list()):
        arm = (
            frame.filter(pl.col("util") == util)
            .group_by("epsilon")
            .agg(pl.col("cost").mean().alias("cost"), pl.col("peak_kw").mean().alias("peak_kw"))
            .sort("epsilon")
        )
        axis.plot(arm["peak_kw"], arm["cost"], marker="o", label=f"util={util:g}")
    axis.set_xlabel("peak kW")
    axis.set_ylabel("mean bill")
    axis.set_title("ε-peak front (20-day window)")
    axis.legend(frameon=False)
    figure.tight_layout()
    dest = REPO_ROOT / "plots" / "pareto_front.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(dest, dpi=140)
    print(f"wrote {dest}")
    return 0


def plot_dlmp() -> int:
    path = REPO_ROOT / "results" / "dlmp_day.csv"
    if not path.exists():
        print(f"missing {path}", file=sys.stderr)
        return 1
    frame = pl.read_csv(path)
    figure, axis = plt.subplots(figsize=(7.2, 3.4))
    axis.plot(range(frame.height), frame["peak_dual"], drawstyle="steps-post", label="peak dual")
    axis.plot(range(frame.height), frame["cap_dual"], drawstyle="steps-post", label="cap dual", alpha=0.6)
    axis.set_xlabel("committed slot")
    axis.set_ylabel("dual")
    axis.set_title("Peak and cap duals, one day, ε=0.01")
    axis.legend(frameon=False)
    figure.tight_layout()
    dest = REPO_ROOT / "plots" / "dlmp_day.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(dest, dpi=140)
    print(f"wrote {dest}")
    return 0


def main() -> int:
    code = plot_front()
    code = plot_dlmp() or code
    return code


if __name__ == "__main__":
    sys.exit(main())
