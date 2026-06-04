"""HotpotQA adapter."""

from __future__ import annotations

from mas_scope.core.types import TaskExample
from mas_scope.datasets.base import BaseDatasetAdapter
from mas_scope.datasets.validators import require_fields


class HotpotQAAdapter(BaseDatasetAdapter):
    dataset_name = "hotpotqa"

    def validate_raw(self, example: dict) -> list[str]:
        return require_fields(example, ["question", "answer"])

    def normalize(self, raw: dict, split: str) -> TaskExample:
        example_id = str(raw.get("_id") or raw.get("id"))
        return TaskExample(
            example_id=example_id,
            dataset_name=self.dataset_name,
            task_type="qa",
            split=split,
            input={
                "question": raw["question"],
                "context": raw.get("context"),
                "supporting_facts": raw.get("supporting_facts"),
            },
            target={
                "answer": raw["answer"],
                "supporting_facts": raw.get("supporting_facts"),
            },
            metadata={"raw_id_field": "_id" if "_id" in raw else "id"},
        )
