"""CH-01, CH-06, CH-08 on one synthetic week. No bank file required."""

from __future__ import annotations

import csv
import sys

from eci.config import REPO_ROOT
from eci.eval.chaos import run_chaos_suite


def main() -> int:
    rows = run_chaos_suite()
    dest = REPO_ROOT / "results" / "chaos_report.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    failed = [row for row in rows if not row["passed"]]
    for row in rows:
        print(f"{row['case']} passed={row['passed']} value={row['cost_delta']} {row['note']}")
    print(f"wrote {dest}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
