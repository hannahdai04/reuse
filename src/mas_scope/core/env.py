"""Minimal environment file loader for local CLI and providers."""

from __future__ import annotations

import os
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _candidate_dirs() -> list[Path]:
    candidates: list[Path] = []
    for path in (Path.cwd(), _repo_root()):
        resolved = path.resolve()
        if resolved not in candidates:
            candidates.append(resolved)
    return candidates


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def load_environment(override: bool = False) -> dict[str, str]:
    """Load environment variables from .env, then .env.example if present."""

    loaded: dict[str, str] = {}
    for directory in _candidate_dirs():
        for filename in (".env", ".env.example"):
            env_path = directory / filename
            if not env_path.exists():
                continue
            for key, value in _parse_env_file(env_path).items():
                if value == "":
                    continue
                if override or key not in os.environ or not os.environ.get(key):
                    os.environ[key] = value
                    loaded[key] = value
    return loaded
