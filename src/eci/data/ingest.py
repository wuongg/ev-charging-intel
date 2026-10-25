"""ACN API pagination and recovery of the truncated Kaggle JSON mirror."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from pathlib import Path

import orjson
import requests

LOGGER = logging.getLogger(__name__)

ACN_API = "https://ev.caltech.edu/api/v1/sessions"
PAGE_SIZE = 25
PAGE_SLEEP_SECONDS = 0.5


def recover_sessions(text: str, *, byte_length: int | None = None) -> tuple[list[dict], bool]:
    """Stream-parse session objects. A truncated tail is logged and skipped.

    The Kaggle mirror is one JSON document whose ``_items`` array is cut off
    mid-file (byte 7,754,984). Complete objects are kept; the partial tail is not.
    """
    marker = text.find('"_items"')
    if marker == -1:
        sessions, truncated = _recover_loose(text)
    else:
        bracket = text.find("[", marker)
        if bracket == -1:
            sessions, truncated = [], True
        else:
            sessions, truncated = _extract_array(text, bracket)
    if truncated:
        nbytes = byte_length if byte_length is not None else len(text.encode("utf-8"))
        LOGGER.warning("truncated at byte %d, recovered %d sessions", nbytes, len(sessions))
    return sessions, truncated


def load_session_mirror(path: Path) -> tuple[list[dict], bool]:
    """Load a raw JSON or JSON.gz mirror, including a truncated one."""
    raw_path = Path(path)
    raw = raw_path.read_bytes()
    if raw_path.suffix == ".gz":
        import gzip

        raw = gzip.decompress(raw)
    text = raw.decode("utf-8", errors="replace")
    return recover_sessions(text, byte_length=len(raw))


def fetch_acn_site(
    site: str,
    token: str,
    start: str,
    end: str,
) -> Iterator[list[dict]]:
    """GET /sessions/{site}, following ``_links.next``. One page is 25 sessions.

    ``start`` and ``end`` are forwarded as a ``where`` clause on connectionTime.
    The caller sleeps between pages (0.5s) inside this generator.
    """
    url: str | None = f"{ACN_API}/{site}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    params: dict | None = {
        "max_results": PAGE_SIZE,
        "sort": "connectionTime",
        "where": f'connectionTime>="{start}" and connectionTime<="{end}"',
    }
    while url:
        response = requests.get(url, headers=headers, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        items = payload.get("_items") or []
        if not isinstance(items, list):
            raise ValueError("ACN payload _items is not a list")
        yield items
        link = (payload.get("_links") or {}).get("next") or {}
        url = link.get("href")
        params = None
        if url:
            time.sleep(PAGE_SLEEP_SECONDS)


def _scan_string(text: str, index: int) -> int:
    """Index just past the closing quote. ``index`` points at the opening quote."""
    index += 1
    limit = len(text)
    while index < limit:
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == '"':
            return index + 1
        index += 1
    return index


def _extract_array(text: str, bracket: int) -> tuple[list[dict], bool]:
    """Parse objects inside the array that opens at ``bracket``. Truncation is not fatal."""
    index = bracket + 1
    limit = len(text)
    sessions: list[dict] = []
    while index < limit:
        while index < limit and text[index] in " \t\r\n,":
            index += 1
        if index >= limit:
            return sessions, True
        if text[index] == "]":
            return sessions, False
        if text[index] != "{":
            return sessions, True
        obj, index, complete = _read_object(text, index)
        if not complete:
            return sessions, True
        if isinstance(obj, dict):
            sessions.append(obj)
    return sessions, True


def _read_object(text: str, start: int) -> tuple[dict | None, int, bool]:
    index = start
    limit = len(text)
    depth = 0
    while index < limit:
        char = text[index]
        if char == '"':
            index = _scan_string(text, index)
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                blob = text[start : index + 1]
                try:
                    loaded = orjson.loads(blob)
                except orjson.JSONDecodeError:
                    return None, index, False
                return loaded, index + 1, True
        index += 1
    return None, index, False


def _recover_loose(text: str) -> tuple[list[dict], bool]:
    stripped = text.lstrip()
    if not stripped:
        return [], False
    if stripped.startswith("["):
        bracket = text.find("[")
        return _extract_array(text, bracket)
    try:
        loaded = orjson.loads(text)
    except orjson.JSONDecodeError:
        start = text.find("{")
        if start == -1:
            return [], True
        obj, _, complete = _read_object(text, start)
        if complete and isinstance(obj, dict):
            return [obj], False
        return [], True
    if isinstance(loaded, dict) and isinstance(loaded.get("_items"), list):
        return [item for item in loaded["_items"] if isinstance(item, dict)], False
    if isinstance(loaded, dict):
        return [loaded], False
    if isinstance(loaded, list):
        return [item for item in loaded if isinstance(item, dict)], False
    return [], False
