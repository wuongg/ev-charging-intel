"""Chronos zero-shot member.

Day 2 does not install chronos-forecasting. The package pulls in torch, which
is a multi-gigabyte environment change and is outside this run. The member is
dropped on purpose; the ensemble is seasonal-naive plus LightGBM.
"""

from __future__ import annotations


def chronos_zero_shot(*_args: object, **_kwargs: object) -> None:
    """Dropped member, graceful by design. Do not install torch to unblock this."""
    raise ImportError(
        "chronos-forecasting is not installed (torch dependency). Dropped member, graceful by design."
    )
