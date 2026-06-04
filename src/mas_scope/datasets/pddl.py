"""PDDL planning task adapter."""

from __future__ import annotations

from mas_scope.core.types import TaskExample
from mas_scope.datasets.base import BaseDatasetAdapter


class PDDLAdapter(BaseDatasetAdapter):
    dataset_name = "pddl"

    def validate_raw(self, example: dict) -> list[str]:
        errors: list[str] = []
        for field in ("domain_pddl", "problem_pddl"):
            if field not in example or not example[field]:
                errors.append(f"missing required field: {field}")
        if not example.get("instruction") and not example.get("goal"):
            errors.append("instruction or goal should exist")
        return errors

    def normalize(self, raw: dict, split: str) -> TaskExample:
        goal = raw.get("goal") or raw.get("instruction")
        return TaskExample(
            example_id=str(raw.get("id")),
            dataset_name=self.dataset_name,
            task_type="formal_planning",
            split=split,
            input={
                "instruction": raw.get("instruction") or goal,
                "domain_pddl": raw["domain_pddl"],
                "problem_pddl": raw["problem_pddl"],
            },
            target={
                "reference_plan": raw.get("reference_plan"),
                "goal": goal,
            },
            metadata={},
        )
