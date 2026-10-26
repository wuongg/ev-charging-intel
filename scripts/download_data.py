"""Mirror sessions to data/raw and record SHA-256 in data/manifest.yaml.

Path A uses ACN_DATA_TOKEN against the Caltech API.
Path B (default today) is the public Kaggle mirror and tolerates the truncated tail.
Path B1 (--source static) sparse-checkouts the ACN-Data-Static garage time series.
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import orjson

from eci.config import REPO_ROOT
from eci.data.download import download_kaggle_mirror, download_twente, download_weather
from eci.data.ingest import fetch_acn_site
from eci.data.manifest import sha256_file, write_manifest_entry

LOGGER = logging.getLogger(__name__)

ACN_START = "2018-09-01"
ACN_END = "2021-12-31"
STATIC_REPO = "https://github.com/tongxin-li/ACN-Data-Static.git"
STATIC_PATHS = (
    "time series data/caltech/California_Garage_01",
    "time series data/jpl/Arroyo_Garage_01",
)


def download_static_snapshot(raw_dir: Path) -> int:
    """Sparse-checkout the two main garages. Skip the clone when files are already there.

    If the two-garage checkout fails, fall back to California_Garage_01 and say so.
    """
    dest = raw_dir / "acn_static_repo"
    caltech = dest / "time series data" / "caltech" / "California_Garage_01"
    jpl = dest / "time series data" / "jpl" / "Arroyo_Garage_01"
    if caltech.exists() and any(caltech.glob("*.csv.gz")):
        jpl_n = sum(1 for _ in jpl.glob("*.csv.gz")) if jpl.exists() else 0
        caltech_n = sum(1 for _ in caltech.glob("*.csv.gz"))
        if jpl_n == 0:
            print("b1_coverage=California_Garage_01 only; JPL was not in the checkout")
        print(f"static_already_present caltech={caltech_n} jpl={jpl_n}")
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not (dest / ".git").exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--sparse",
                "--depth",
                "1",
                STATIC_REPO,
                str(dest),
            ],
            check=True,
        )
    try:
        subprocess.run(["git", "sparse-checkout", "set", *STATIC_PATHS], cwd=dest, check=True)
    except subprocess.CalledProcessError:
        LOGGER.warning("two-garage sparse-checkout failed; keeping California_Garage_01 only")
        subprocess.run(
            ["git", "sparse-checkout", "set", STATIC_PATHS[0]],
            cwd=dest,
            check=True,
        )
        print("b1_coverage=California_Garage_01 only; JPL checkout failed")
    caltech_n = sum(1 for _ in caltech.glob("*.csv.gz")) if caltech.exists() else 0
    jpl_n = sum(1 for _ in jpl.glob("*.csv.gz")) if jpl.exists() else 0
    print(f"static_files caltech={caltech_n} jpl={jpl_n}")
    if caltech_n <= 0:
        LOGGER.error("static snapshot recovered 0 California sessions")
        return 1
    return 0


def _download_acn(site: str, token: str, raw_dir: Path) -> dict:
    pages = []
    total = 0
    for page in fetch_acn_site(site, token, ACN_START, ACN_END):
        pages.append(page)
        total += len(page)
        LOGGER.info("acn page sessions=%d total=%d", len(page), total)
    items = [session for page in pages for session in page]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    relative = f"data/raw/acn_{site}_{stamp}.json.gz"
    target = REPO_ROOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = orjson.dumps({"_items": items, "_meta": {"site": site, "start": ACN_START, "end": ACN_END}})
    with gzip.open(target, "wb") as handle:
        handle.write(payload)
    return {
        "file": relative.replace("\\", "/"),
        "sha256": sha256_file(target),
        "source_url": f"https://ev.caltech.edu/api/v1/sessions/{site}",
        "downloaded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rows_recovered": len(items),
        "truncated": False,
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Mirror ACN sessions and optional KBUR weather")
    parser.add_argument("--source", choices=["auto", "kaggle", "acn", "static", "twente"], default="auto")
    parser.add_argument("--site", default="caltech")
    args = parser.parse_args(argv)

    raw_dir = REPO_ROOT / "data" / "raw"
    if args.source == "twente":
        entry = download_twente(raw_dir)
        if entry is None:
            LOGGER.error("Twente download failed (CC BY-NC-SA). Drift demo will use the ACN 2018-vs-2020 proxy.")
            return 1
        write_manifest_entry(entry)
        print("license=CC BY-NC-SA 4.0 demo-only not-redistributed")
        print(f"file={entry['file']}")
        return 0
    if args.source == "static":
        return download_static_snapshot(raw_dir)
    token = os.environ.get("ACN_DATA_TOKEN", "").strip()
    source = args.source
    if source == "auto":
        source = "acn" if token else "kaggle"

    entry: dict | None = None
    if source == "acn":
        if not token:
            LOGGER.error("ACN_DATA_TOKEN is empty")
            return 1
        try:
            entry = _download_acn(args.site, token, raw_dir)
        except Exception as exc:  # noqa: BLE001 — fallback is the day-1 contract
            LOGGER.warning("ACN API failed (%s); falling back to the Kaggle mirror", exc)
            entry = None
    if entry is None:
        entry = download_kaggle_mirror(raw_dir)

    write_manifest_entry(entry)
    print(f"sessions_recovered={entry['rows_recovered']}")
    print(f"truncated={entry['truncated']}")
    print(f"sha256={entry['sha256']}")
    print(f"file={entry['file']}")
    if entry["rows_recovered"] <= 0:
        LOGGER.error("recovered 0 sessions")
        return 1

    weather = download_weather(raw_dir)
    if weather is not None:
        write_manifest_entry(weather)
        print(f"weather_rows={weather['rows_recovered']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
