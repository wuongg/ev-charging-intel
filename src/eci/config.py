"""Load YAML configs and stamp provenance on every artifact."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "data" / "manifest.yaml"


class Settings(BaseModel):
    """Loose YAML document. Nested keys stay dicts; tariff files use TariffConfig."""

    model_config = ConfigDict(extra="allow")


def load_config(path: Path) -> Settings:
    """Load one YAML file into a pydantic Settings model."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return Settings.model_validate(raw)


def config_hash(cfg: Settings) -> str:
    """SHA-256 of the config rendered as normalized YAML (sorted keys)."""
    payload = cfg.model_dump(mode="json")
    canonical = yaml.safe_dump(payload, sort_keys=True, allow_unicode=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        status = ""
    if status.strip():
        return f"{sha}-dirty"
    return sha


def provenance(config_path: Path | None = None) -> dict[str, str]:
    """Identity of a generated artifact.

    data_hash is the hash of data/manifest.yaml (missing until the first download).
    """
    from eci.data.manifest import compute_data_hash

    cfg_hash = ""
    if config_path is not None:
        cfg_hash = config_hash(load_config(config_path))
    return {
        "git_sha": _git_sha(),
        "config_hash": cfg_hash,
        "data_hash": compute_data_hash(MANIFEST_PATH),
        "utc_now": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
