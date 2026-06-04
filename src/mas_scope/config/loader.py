"""YAML experiment config loader."""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

from mas_scope.config.schema import ExperimentConfig
from mas_scope.core.env import load_environment


ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(value):
    if isinstance(value, str):
        return ENV_PATTERN.sub(lambda match: os.getenv(match.group(1)) or match.group(0), value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    load_environment()
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return ExperimentConfig.model_validate(_expand_env(payload))
