"""SHA-256 manifest of on-disk snapshots. The manifest hash is data_hash."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from eci.config import MANIFEST_PATH


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest_entry(entry: dict, manifest_path: Path | None = None) -> Path:
    """Insert or replace the entry with the same ``file`` key and rewrite YAML."""
    path = Path(manifest_path) if manifest_path is not None else MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    files: list[dict] = []
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        files = list(loaded.get("files") or [])
    files = [item for item in files if item.get("file") != entry.get("file")]
    files.append(entry)
    files.sort(key=lambda item: str(item.get("file", "")))
    document = {"files": files}
    path.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def compute_data_hash(manifest_path: Path | None = None) -> str:
    """SHA-256 of the manifest's file list. ``missing`` when no manifest exists."""
    path = Path(manifest_path) if manifest_path is not None else MANIFEST_PATH
    if not path.exists():
        return "missing"
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    files = loaded.get("files") or []
    canonical = json.dumps(files, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
