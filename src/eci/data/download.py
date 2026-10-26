"""Optional weather download and the Kaggle session mirror (no token)."""

from __future__ import annotations

import logging
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

from eci.data.ingest import load_session_mirror
from eci.data.manifest import sha256_file

LOGGER = logging.getLogger(__name__)

TWENTE_PAGE = "https://data.4tu.nl/datasets/80ef3824-3f5d-4e45-8794-3b8791efbd13/1"
TWENTE_API = "https://data.4tu.nl/v2/articles?search_for=80ef3824-3f5d-4e45-8794-3b8791efbd13"
TWENTE_LICENSE = "CC BY-NC-SA 4.0"

KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/dextersebs/ev-dataset"
IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
USER_AGENT = "eci-day1"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stream_to(url: str, dest: Path, params: list[tuple[str, str]] | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".part")
    with requests.get(
        url,
        params=params,
        stream=True,
        timeout=180,
        headers={"User-Agent": USER_AGENT},
    ) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(1 << 16):
                if chunk:
                    handle.write(chunk)
    temporary.replace(dest)


def download_kaggle_mirror(raw_dir: Path) -> dict:
    """Download the Kaggle zip and extract acndata_sessions.json.

    The JSON is truncated. Recovery happens in ``load_session_mirror`` and does
    not raise on the broken tail.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "kaggle_ev_dataset.zip"
    _stream_to(KAGGLE_URL, archive)
    with zipfile.ZipFile(archive) as bundle:
        names = [name for name in bundle.namelist() if name.endswith("acndata_sessions.json")]
        if not names:
            raise FileNotFoundError("acndata_sessions.json not in the Kaggle zip")
        member = names[0]
        target = raw_dir / "acndata_sessions.json"
        target.write_bytes(bundle.read(member))
    sessions, truncated = load_session_mirror(target)
    return {
        "file": "data/raw/acndata_sessions.json",
        "sha256": sha256_file(target),
        "source_url": KAGGLE_URL,
        "downloaded_at": _now(),
        "rows_recovered": len(sessions),
        "truncated": bool(truncated),
    }


def download_weather(raw_dir: Path) -> dict | None:
    """NOAA/IEM ASOS for Burbank KBUR, 2018–2021 hourly. Failure is a warning.

    Weather is optional and must not block the session mirror.
    """
    dest = raw_dir / "weather" / "kbur_asos_2018_2021.csv"
    params = [
        ("station", "KBUR"),
        ("data", "tmpf"),
        ("data", "relh"),
        ("data", "drct"),
        ("data", "sped"),
        ("data", "mslp"),
        ("network", "CA_ASOS"),
        ("year1", "2018"),
        ("month1", "1"),
        ("day1", "1"),
        ("year2", "2021"),
        ("month2", "12"),
        ("day2", "31"),
        ("tz", "Etc/UTC"),
        ("format", "onlycomma"),
        ("latlon", "no"),
        ("elev", "no"),
        ("missing", "M"),
        ("trace", "T"),
        ("direct", "no"),
        ("report_type", "3"),
        ("report_type", "4"),
    ]
    try:
        try:
            _stream_to(IEM_ASOS_URL, dest, params)
        except requests.RequestException:
            LOGGER.warning("weather request with network=CA_ASOS failed; retrying without network")
            _stream_to(IEM_ASOS_URL, dest, [item for item in params if item[0] != "network"])
    except requests.RequestException as exc:
        LOGGER.warning("weather download failed (optional, not blocking): %s", exc)
        return None
    head = dest.read_text(encoding="utf-8", errors="replace")[:500].lower()
    if "tmpf" not in head and "station" not in head:
        LOGGER.warning("weather download did not look like ASOS CSV; skipping manifest entry")
        return None
    rows = max(sum(1 for _ in dest.open(encoding="utf-8", errors="replace")) - 1, 0)
    return {
        "file": "data/raw/weather/kbur_asos_2018_2021.csv",
        "sha256": sha256_file(dest),
        "source_url": IEM_ASOS_URL,
        "downloaded_at": _now(),
        "rows_recovered": rows,
        "truncated": False,
    }


def download_twente(raw_dir: Path, timeout: int = 20) -> dict | None:
    """Fetch the a.s.r. office-parking workbook from 4TU. NC license; demo only.

    Returns None when 4TU is unreachable. The drift demo then uses the ACN
    2018-vs-2020 proxy and labels the CSV as a fallback.
    """
    dest_dir = Path(raw_dir) / "twente"
    dest_dir.mkdir(parents=True, exist_ok=True)
    marker = dest_dir / "LICENSE_CC_BY-NC-SA.txt"
    marker.write_text(
        "CC BY-NC-SA 4.0. University of Twente / a.s.r. living lab. "
        "DOI 10.4121/80ef3824-3f5d-4e45-8794-3b8791efbd13. "
        "Do not redistribute. Drift-demo use only.\n",
        encoding="utf-8",
    )
    try:
        response = requests.get(
            TWENTE_API,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        LOGGER.warning("Twente 4TU listing failed (%s); not redistributing anything", exc)
        return None
    articles = payload if isinstance(payload, list) else [payload]
    files = []
    for article in articles:
        files.extend(article.get("files") or [])
    if not files:
        LOGGER.warning("Twente 4TU listing had no files")
        return None
    saved = 0
    for item in files:
        url = item.get("download_url") or item.get("url")
        name = item.get("name") or f"twente_{saved}"
        if not url:
            continue
        target = dest_dir / Path(str(name)).name
        try:
            _stream_to(url, target)
            saved += 1
        except requests.RequestException as exc:
            LOGGER.warning("Twente file %s failed: %s", name, exc)
    if saved <= 0:
        return None
    return {
        "file": "data/raw/twente/",
        "sha256": sha256_file(marker),
        "source_url": TWENTE_PAGE,
        "downloaded_at": _now(),
        "rows_recovered": saved,
        "truncated": False,
        "license": TWENTE_LICENSE,
        "redistribute": False,
    }
