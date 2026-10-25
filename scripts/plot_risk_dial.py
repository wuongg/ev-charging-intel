"""Two axes from results/risk_dial.csv. No numbers are typed in."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import polars as pl

from eci.config import REPO_ROOT


def main() -> int:
    path = REPO_ROOT / "results" / "risk_dial.csv"
    if not path.exists():
        print(f"missing {path}", file=sys.stderr)
        return 1
    means = (
        pl.read_csv(path)
        .group_by("tau")
        .agg(
            pl.col("total_cost").mean().alias("cost"),
            pl.col("unmet_pct").mean().alias("unmet"),
            pl.col("penalty_share").mean().alias("share"),
        )
        .sort("tau")
    )
    figure, axis = plt.subplots(figsize=(6.2, 4.0))
    other = axis.twinx()
    axis.plot(means["tau"].to_list(), means["cost"].to_list(), "o-", color="C0", label="total cost")
    other.plot(means["tau"].to_list(), means["unmet"].to_list(), "s--", color="C1", label="unmet %")
    axis.set_xlabel("dwell quantile τ")
    axis.set_ylabel("mean total cost")
    other.set_ylabel("mean unmet %")
    share = float(means["share"].mean())
    axis.set_title(f"SCE, headline M, mean penalty share {share:.1%}")
    lines_a, labels_a = axis.get_legend_handles_labels()
    lines_b, labels_b = other.get_legend_handles_labels()
    axis.legend(lines_a + lines_b, labels_a + labels_b, loc="best")
    figure.tight_layout()
    dest = REPO_ROOT / "plots" / "risk_dial.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(dest, dpi=140)
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
