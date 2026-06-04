"""Lightweight planning metrics that do not require an external planner."""

from __future__ import annotations

from mas_scope.tools.pddl_validator import (
    no_variables_in_plan,
    plan_uses_declared_objects,
    plan_uses_known_actions,
    split_plan,
    valid_action_format,
)


def compute_planning_metrics(prediction, target: dict) -> dict:
    actions = split_plan(prediction)
    metrics = {
        "non_empty_plan": bool(actions),
        "valid_action_format": valid_action_format(prediction),
        "no_variables": no_variables_in_plan(prediction),
        "plan_length": len(actions),
    }
    if "allowed_action_names" in target:
        metrics["uses_known_actions"] = plan_uses_known_actions(prediction, target.get("allowed_action_names"))
    if "declared_objects" in target:
        metrics["uses_declared_objects"] = plan_uses_declared_objects(prediction, target.get("declared_objects"))
    if target.get("reference_plan") is not None:
        metrics["exact_plan_match"] = split_plan(prediction) == split_plan(target.get("reference_plan"))
    return metrics
