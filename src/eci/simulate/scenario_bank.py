"""K synthetic years saved once, then reused by every policy (common random numbers)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from eci.data.manifest import sha256_file
from eci.simulate.bootstrap import day_block_bootstrap


def make_scenario_bank(
    sessions: pl.DataFrame,
    dest: Path,
    *,
    k: int = 50,
    seed: int = 7,
    regime_start: date,
    regime_end: date,
) -> dict:
    """Write the bank and return its sha256 plus sessions-per-year summary."""
    bank = day_block_bootstrap(
        sessions,
        regime_start=regime_start,
        regime_end=regime_end,
        n_years=k,
        seed=seed,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    bank.write_parquet(dest)
    counts = bank.group_by("scenario_id").len()
    mean_sessions = float(counts["len"].mean()) if counts.height else 0.0
    return {
        "path": dest,
        "sha256": sha256_file(dest),
        "k": k,
        "seed": seed,
        "n_rows": bank.height,
        "mean_sessions_per_year": mean_sessions,
        "regime_start": regime_start.isoformat(),
        "regime_end": regime_end.isoformat(),
    }
