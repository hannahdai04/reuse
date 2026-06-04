"""StrategyQA adapter."""

from __future__ import annotations

from mas_scope.core.types import TaskExample
from mas_scope.datasets.base import BaseDatasetAdapter
from mas_scope.datasets.validators import normalize_yes_no, require_fields


class StrategyQAAdapter(BaseDatasetAdapter):
    dataset_name = "strategyqa"

    def validate_raw(self, example: dict) -> list[str]:
        errors = require_fields(example, ["question", "answer"])
        if "answer" in example and normalize_yes_no(example.get("answer")) is None:
            errors.append("answer must be convertible to yes/no")
        return errors

    def normalize(self, raw: dict, split: str) -> TaskExample:
        normalized = normalize_yes_no(raw["answer"])
        example_id = str(raw.get("qid") or raw.get("id"))
        return TaskExample(
            example_id=example_id,
            dataset_name=self.dataset_name,
            task_type="qa",
            split=split,
            input={
                "question": raw["question"],
                "facts": raw.get("facts"),
                "decomposition": raw.get("decomposition"),
                "evidence": raw.get("evidence"),
            },
            target={"answer": raw["answer"], "normalized_answer": normalized},
            metadata={"raw_id_field": "qid" if "qid" in raw else "id"},
        )
