"""One synthetic csv.gz stands in for a B1 session file."""

from __future__ import annotations

import gzip
from pathlib import Path

from eci.data.ingest_static import parse_session_file


def test_parse_session_file_iced_and_pilot(tmp_path: Path) -> None:
    path = tmp_path / "2-39-123-23-2019-01-02T15-00-00-000000.csv.gz"
    rows = [
        ",Charging Current (A),Actual Pilot (A),Voltage (V),Charging State,Energy Delivered (kWh),Power (kW)",
        "2019-01-02T15:00:00+00:00,0.0,32.0,208,READY,0.0,0.0",
        "2019-01-02T15:00:30+00:00,16.0,32.0,208,CHARGING,0.5,3.3",
        "2019-01-02T15:01:00+00:00,16.0,32.0,208,CHARGING,1.2,3.3",
    ]
    with gzip.open(path, "wt", newline="") as handle:
        handle.write("\n".join(rows) + "\n")
    record, profiles = parse_session_file(path, "caltech", "California_Garage_01")
    assert record["stationID"] == "2-39-123-23"
    assert record["siteID"] == "0002"
    assert record["kWhDelivered"] == 1.2
    assert record["userInputs"] is None
    assert record["extra"]["pilot_amp"] == 32.0
    # 30s at 0 A out of 60s of counted gaps.
    assert record["extra"]["iced_fraction"] == 0.5
    assert len(profiles) == 2
    assert profiles[0]["current_a"] == 8.0
