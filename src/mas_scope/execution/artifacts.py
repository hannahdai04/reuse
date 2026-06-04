"""Run artifact writer."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import BaseModel


def _jsonable(value):
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


class ArtifactWriter:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("examples.jsonl", "trajectories.jsonl", "results.jsonl", "errors.jsonl", "environment_steps.jsonl"):
            (self.run_dir / filename).write_text("", encoding="utf-8")

    def write_yaml(self, filename: str, payload: dict) -> None:
        with (self.run_dir / filename).open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)

    def write_json(self, filename: str, payload: dict) -> None:
        with (self.run_dir / filename).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def append_jsonl(self, filename: str, payload) -> None:
        with (self.run_dir / filename).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(payload), ensure_ascii=False) + "\n")
