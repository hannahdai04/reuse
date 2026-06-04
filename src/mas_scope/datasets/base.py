"""Base dataset adapter and shared loading logic."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

from mas_scope.core.exceptions import DatasetValidationError
from mas_scope.core.types import TaskExample


class BaseDatasetAdapter(ABC):
    dataset_name: str

    def __init__(self, data_path: str | Path) -> None:
        self.data_path = Path(data_path)

    def load(self, split: str, limit: int | None = None) -> list[TaskExample]:
        records = self._read_records()
        if limit is not None:
            records = records[:limit]
        examples: list[TaskExample] = []
        all_errors: list[str] = []
        for index, raw in enumerate(records):
            errors = self.validate_raw(raw)
            if errors:
                all_errors.extend([f"record {index}: {error}" for error in errors])
                continue
            examples.append(self.normalize(raw, split))
        if all_errors:
            raise DatasetValidationError("; ".join(all_errors))
        return examples

    def _read_records(self) -> list[dict]:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
        if self.data_path.suffix.lower() == ".jsonl":
            records: list[dict] = []
            with self.data_path.open("r", encoding="utf-8-sig") as handle:
                for line in handle:
                    stripped = line.strip()
                    if stripped:
                        records.append(json.loads(stripped))
            return records

        with self.data_path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            return payload["data"]
        if isinstance(payload, dict):
            return [payload]
        raise DatasetValidationError(f"Unsupported JSON payload in {self.data_path}")

    @abstractmethod
    def validate_raw(self, example: dict) -> list[str]:
        ...

    @abstractmethod
    def normalize(self, raw: dict, split: str) -> TaskExample:
        ...
