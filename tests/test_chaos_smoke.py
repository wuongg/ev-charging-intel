"""CH-01, CH-06, and CH-08 on the synthetic week used by CI."""

from eci.eval.chaos import run_chaos_suite


def test_chaos_suite_passes() -> None:
    rows = run_chaos_suite()
    assert [row["case"] for row in rows] == ["CH-01", "CH-06", "CH-08"]
    failed = [row for row in rows if not row["passed"]]
    assert failed == []
