"""Golden tariff day and schema checks for the filed SCE and EVN rate cards."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from eci.config import config_hash, load_config, provenance
from eci.data.validate_tariff import (
    TariffConfig,
    TariffWindow,
    load_tariff,
    tariff_profile,
    validate_tariff,
)

ROOT = Path(__file__).resolve().parents[1]
SCE = ROOT / "configs" / "tariff_sce_ev4.yaml"
EVN = ROOT / "configs" / "tariff_evn_2026.yaml"
SITE = ROOT / "configs" / "site_caltech.yaml"
GOLDEN = ROOT / "tests" / "golden" / "tariff_golden_2019-01-15_sce.json"


def _covered_week() -> TariffConfig:
    days = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    return TariffConfig(
        timezone="America/Los_Angeles",
        currency="USD",
        windows=[
            TariffWindow(name="off_peak", start="00:00", end="08:00", days=days, price_per_kwh=0.1),
            TariffWindow(name="normal", start="08:00", end="16:00", days=days, price_per_kwh=0.2),
            TariffWindow(name="peak", start="16:00", end="24:00", days=days, price_per_kwh=0.4),
        ],
    )


def test_golden_sce_day_is_frozen() -> None:
    cfg = load_tariff(SCE)
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    profile = tariff_profile(cfg, date(2019, 1, 15))
    assert profile == golden["prices_per_kwh"]
    assert len(profile) == 24


def test_shipped_tariffs_and_site_load() -> None:
    validate_tariff(load_tariff(SCE))
    evn = load_tariff(EVN)
    validate_tariff(evn)
    sce = load_tariff(SCE)
    site = load_config(SITE).model_dump()
    assert site["evse_count"] == 54
    assert site["dcfc_count"] == 1
    assert site["dcfc_power_kw"] == 50
    assert site["site_cap_kw"] == 150
    assert site["timezone"] == "America/Los_Angeles"
    assert site["evse_power_default_kw"] == 6.7
    assert sce.demand_charge_per_kw_month == 15.51
    summer = tariff_profile(sce, date(2019, 7, 16))
    assert summer[12] == pytest.approx(0.26668)
    assert summer[0] == pytest.approx(0.05623)


def test_evn_peak_bounds_are_q963() -> None:
    cfg = load_tariff(EVN)
    peaks = [window for window in cfg.windows if window.name == "peak"]
    assert len(peaks) == 1
    assert peaks[0].start == "17:30"
    assert peaks[0].end == "22:30"
    assert "sun" not in peaks[0].days
    assert set(peaks[0].days) == {"mon", "tue", "wed", "thu", "fri", "sat"}
    cfg = load_tariff(EVN)
    tiers = {}
    for window in cfg.windows:
        tiers.setdefault(window.name, window.price_per_kwh)
    sunday = tariff_profile(cfg, date(2019, 1, 13))
    tuesday = tariff_profile(cfg, date(2019, 1, 15))
    assert all(price != tiers["peak"] for price in sunday)
    assert tuesday[0] == tiers["off_peak"]
    assert tuesday[5] == tiers["off_peak"]
    assert tuesday[6] == tiers["normal"]
    assert tuesday[17] == tiers["normal"]
    assert tuesday[18] == tiers["peak"]
    assert tuesday[22] == tiers["peak"]
    assert tuesday[23] == tiers["normal"]


def test_overlap_gap_price_and_order_fail() -> None:
    overlapping = _covered_week()
    overlapping.windows.append(
        TariffWindow(name="peak", start="10:00", end="12:00", days=["mon"], price_per_kwh=0.4)
    )
    with pytest.raises(ValueError, match="overlap"):
        validate_tariff(overlapping)

    gapped = _covered_week()
    gapped.windows[2] = TariffWindow(
        name="peak",
        start="16:00",
        end="20:00",
        days=["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
        price_per_kwh=0.4,
    )
    with pytest.raises(ValueError, match="gap"):
        validate_tariff(gapped)

    free = _covered_week()
    free.windows[0] = TariffWindow(
        name="off_peak",
        start="00:00",
        end="08:00",
        days=["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
        price_per_kwh=0,
    )
    with pytest.raises(ValueError, match="price"):
        validate_tariff(free)

    inverted = _covered_week()
    inverted.windows[0] = TariffWindow(
        name="off_peak",
        start="00:00",
        end="08:00",
        days=["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
        price_per_kwh=0.5,
    )
    with pytest.raises(ValueError, match="order"):
        validate_tariff(inverted)


def test_nonpositive_demand_and_implausible_price_fail() -> None:
    sce = load_tariff(SCE)
    sce.demand_charge_per_kw_month = 0
    with pytest.raises(ValueError, match="demand"):
        validate_tariff(sce)
    wild = _covered_week()
    wild.currency = "USD"
    wild.windows[0] = TariffWindow(
        name="off_peak",
        start="00:00",
        end="08:00",
        days=["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
        price_per_kwh=10.0,
    )
    with pytest.raises(ValueError, match="plausibility"):
        validate_tariff(wild)


def test_config_hash_and_provenance_keys() -> None:
    cfg = load_config(SCE)
    assert config_hash(cfg) == config_hash(load_config(SCE))
    assert len(config_hash(cfg)) == 64
    prov = provenance(SCE)
    assert set(prov) == {"git_sha", "config_hash", "data_hash", "utc_now"}
    assert prov["data_hash"] == "missing" or len(prov["data_hash"]) == 64
