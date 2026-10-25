"""Tariff schema: full-week coverage, no overlap, ordered prices, hourly profile."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from pydantic import BaseModel, ConfigDict, Field

DAY_NAMES = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")

_DAY_ALIASES = {
    "mon": "mon",
    "monday": "mon",
    "t2": "mon",
    "0": "mon",
    "tue": "tue",
    "tuesday": "tue",
    "t3": "tue",
    "1": "tue",
    "wed": "wed",
    "wednesday": "wed",
    "t4": "wed",
    "2": "wed",
    "thu": "thu",
    "thursday": "thu",
    "t5": "thu",
    "3": "thu",
    "fri": "fri",
    "friday": "fri",
    "t6": "fri",
    "4": "fri",
    "sat": "sat",
    "saturday": "sat",
    "t7": "sat",
    "5": "sat",
    "sun": "sun",
    "sunday": "sun",
    "cn": "sun",
    "6": "sun",
}

_TIER_ALIASES = {
    "off_peak": "off_peak",
    "offpeak": "off_peak",
    "off": "off_peak",
    "normal": "normal",
    "mid": "normal",
    "mid_peak": "normal",
    "shoulder": "normal",
    "peak": "peak",
    "on_peak": "peak",
    "onpeak": "peak",
}


class TariffWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    start: str
    end: str
    days: list[str]
    price_per_kwh: float
    months: list[int] | None = None


class TariffConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    timezone: str
    currency: str
    windows: list[TariffWindow] = Field(min_length=1)
    rate_status: str | None = None
    demand_charge_per_kw_month: float | None = None


_PLAUSIBLE = {
    "USD": (0.01, 2.0),
    "VND": (100.0, 20_000.0),
}
_DEMAND_PLAUSIBLE = {
    "USD": (0.0, 80.0),
    "VND": (0.0, 1_000_000.0),
}


def canonical_day(value: str) -> str:
    key = str(value).strip().lower()
    try:
        return _DAY_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"unknown day {value!r}") from exc


def canonical_tier(name: str) -> str:
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    return _TIER_ALIASES.get(key, key)


def parse_hhmm(value: str) -> int:
    """Minutes since midnight. '24:00' is the exclusive end of the civil day."""
    text = str(value).strip()
    parts = text.split(":")
    if len(parts) < 2:
        raise ValueError(f"clock time must be HH:MM, got {value!r}")
    hour = int(parts[0])
    minute = int(parts[1])
    if text in {"24:00", "24:00:00"}:
        return 24 * 60
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError(f"clock time out of range: {value!r}")
    return hour * 60 + minute


def _window_months(window: TariffWindow) -> range | list[int]:
    if window.months is None:
        return list(range(1, 13))
    return window.months


def _price_band(currency: str) -> tuple[float, float] | None:
    return _PLAUSIBLE.get(currency.strip().upper())


def validate_tariff(cfg: TariffConfig) -> None:
    """Raise ValueError when windows overlap, leave a gap, or break price order."""
    errors: list[str] = []
    if not cfg.currency or not str(cfg.currency).strip():
        errors.append("price: currency is empty")
    try:
        ZoneInfo(cfg.timezone)
    except Exception:
        errors.append(f"timezone: {cfg.timezone!r} is not an IANA zone")

    band = _price_band(cfg.currency)
    if cfg.demand_charge_per_kw_month is not None:
        demand = cfg.demand_charge_per_kw_month
        demand_band = _DEMAND_PLAUSIBLE.get(cfg.currency.strip().upper())
        if not isinstance(demand, (int, float)) or isinstance(demand, bool) or demand <= 0:
            errors.append("demand_charge must be > 0")
        elif demand_band is not None and not (demand_band[0] < demand <= demand_band[1]):
            errors.append(f"demand_charge {demand} outside plausibility band {demand_band}")

    seasonal = any(window.months is not None for window in cfg.windows)
    month_groups: list[int | None] = list(range(1, 13)) if seasonal else [None]

    for month in month_groups:
        covered = {day: [False] * (24 * 60) for day in DAY_NAMES}
        tier_prices: dict[str, list[float]] = {"off_peak": [], "normal": [], "peak": []}
        label = "all" if month is None else f"month {month}"
        for window in cfg.windows:
            if month is not None and month not in _window_months(window):
                continue
            if window.months is not None:
                for item in window.months:
                    if item < 1 or item > 12:
                        errors.append(f"months: {window.name} has month {item}")
            price = window.price_per_kwh
            if isinstance(price, bool) or not isinstance(price, (int, float)) or price <= 0:
                errors.append(f"price: {window.name} price_per_kwh must be > 0")
            elif band is not None and not (band[0] <= float(price) <= band[1]):
                errors.append(f"price: {window.name} {price} outside plausibility band {band}")
            tier = canonical_tier(window.name)
            if tier in tier_prices and isinstance(price, (int, float)) and not isinstance(price, bool):
                tier_prices[tier].append(float(price))
            try:
                start = parse_hhmm(window.start)
                end = parse_hhmm(window.end)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if end <= start:
                errors.append(
                    f"overlap: {window.name} must not cross or invert midnight "
                    f"({window.start}->{window.end}); split the window"
                )
                continue
            try:
                days = [canonical_day(day) for day in window.days]
            except ValueError as exc:
                errors.append(str(exc))
                continue
            for day in days:
                for minute in range(start, end):
                    if covered[day][minute]:
                        errors.append(f"overlap: {label} {day} minute {minute} covered twice ({window.name})")
                        break
                    covered[day][minute] = True
        for day, slots in covered.items():
            if not all(slots):
                missing = slots.index(False)
                errors.append(f"gap: {label} {day} missing minute {missing}")
        for tier, prices in tier_prices.items():
            if not prices:
                errors.append(f"order: {label} missing tier {tier}")
        if tier_prices["off_peak"] and tier_prices["normal"] and tier_prices["peak"]:
            off_max = max(tier_prices["off_peak"])
            normal_min = min(tier_prices["normal"])
            normal_max = max(tier_prices["normal"])
            peak_min = min(tier_prices["peak"])
            if not (off_max < normal_min <= normal_max < peak_min):
                errors.append(
                    f"order: {label} require off-peak < normal < peak "
                    f"(off<={off_max}, normal=[{normal_min}, {normal_max}], peak>={peak_min})"
                )

    if errors:
        unique = list(dict.fromkeys(errors))
        raise ValueError("; ".join(unique))


def load_tariff(path: Path) -> TariffConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg = TariffConfig.model_validate(raw)
    validate_tariff(cfg)
    return cfg


def tariff_profile(cfg: TariffConfig, day: date) -> list[float]:
    """Hourly price (length 24) for a civil date in the tariff's own calendar.

    The price of hour h is the window that contains h:00. Windows are half-open,
    so 17:30 belongs to hour 17's start (still the previous window) and hour 18
    sees 18:00.
    """
    validate_tariff(cfg)
    day_name = DAY_NAMES[day.weekday()]
    prices: list[float] = []
    for hour in range(24):
        minute = hour * 60
        match: float | None = None
        for window in cfg.windows:
            days = {canonical_day(item) for item in window.days}
            if day_name not in days:
                continue
            if window.months is not None and day.month not in window.months:
                continue
            start = parse_hhmm(window.start)
            end = parse_hhmm(window.end)
            if start <= minute < end:
                match = float(window.price_per_kwh)
                break
        if match is None:
            raise ValueError(f"gap: {day_name} hour {hour:02d}")
        prices.append(match)
    return prices
