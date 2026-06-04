"""Interactive environment metrics."""

from __future__ import annotations

from mas_scope.core.types import Trajectory


def compute_interactive_metrics(trajectory: Trajectory) -> dict:
    steps = trajectory.environment_steps
    invalid = sum(1 for step in steps if step.info.get("invalid_action"))
    success = bool(steps and steps[-1].info.get("success"))
    stopped_by_max_steps = bool(steps and steps[-1].info.get("stopped_by_max_steps"))
    return {
        "success": success,
        "final_score": steps[-1].reward if steps else 0.0,
        "steps": len(steps),
        "invalid_action_rate": invalid / len(steps) if steps else 0.0,
        "stopped_by_max_steps": stopped_by_max_steps,
    }
