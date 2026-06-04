"""Formal planning task."""

from __future__ import annotations

from mas_scope.core.types import AgentSpec, TaskExample, Trajectory
from mas_scope.evaluation.planning_metrics import compute_planning_metrics
from mas_scope.tasks.base import BaseTask
from mas_scope.tools.pddl_validator import (
    clean_plan_output,
    extract_action_names,
    extract_declared_objects,
    format_action_signatures,
    format_declared_objects,
)


class FormalPlanningTask(BaseTask):
    task_type = "formal_planning"

    def build_prompt(self, example: TaskExample, agent_spec: AgentSpec, context: dict) -> str:
        domain_pddl = example.input.get("domain_pddl")
        problem_pddl = example.input.get("problem_pddl")
        action_text = format_action_signatures(domain_pddl)
        object_text = format_declared_objects(problem_pddl, domain_pddl)
        memory_section = self._memory_section(context)
        return (
            f"Instruction: {example.input.get('instruction')}\n"
            f"Goal:\n{example.target.get('goal')}\n"
            f"Allowed action signatures: {action_text}\n"
            f"Declared objects/constants: {object_text}\n"
            f"Domain PDDL:\n{domain_pddl}\n"
            f"Problem PDDL:\n{problem_pddl}\n"
            f"{memory_section}"
            "Planning output protocol:\n"
            "- This is a planning task, not a domain modeling task.\n"
            "- Do not include reasoning or explanation.\n"
            "- Do not use markdown, code fences, bullets, numbering, JSON, or YAML.\n"
            "- Do not output PDDL schema sections such as :action, :parameters, :precondition, or :effect.\n"
            "- Do not output a PDDL domain file or a PDDL problem file.\n"
            "- Return only grounded plan actions, one parenthesized action per line.\n"
            "- Each plan action must use one allowed action name and only declared objects/constants.\n"
            "- Never use variables such as ?x, ?from, or ?to in the final plan.\n"
            "- Example format: (pick-up a)\n"
            "- Stop after the final action."
        )

    def parse_final_answer(self, trajectory: Trajectory):
        if trajectory.final_answer is not None:
            return clean_plan_output(trajectory.final_answer)
        return clean_plan_output(trajectory.messages[-1].content.strip()) if trajectory.messages else None

    def evaluate(self, prediction, example: TaskExample) -> dict:
        target = dict(example.target)
        domain_pddl = example.input.get("domain_pddl")
        problem_pddl = example.input.get("problem_pddl")
        target["allowed_action_names"] = extract_action_names(domain_pddl)
        target["declared_objects"] = extract_declared_objects(problem_pddl, domain_pddl)
        return compute_planning_metrics(prediction, target)
