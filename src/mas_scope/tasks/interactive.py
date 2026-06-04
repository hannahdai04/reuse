"""Interactive task loop."""

from __future__ import annotations

from mas_scope.core.ids import make_run_id
from mas_scope.core.types import AgentSpec, EnvironmentStep, TaskExample, Trajectory
from mas_scope.environments.base import BaseEnvironmentAdapter
from mas_scope.evaluation.interactive_metrics import compute_interactive_metrics
from mas_scope.tasks.base import BaseTask


class InteractiveTask(BaseTask):
    task_type = "interactive"

    def __init__(self, environment: BaseEnvironmentAdapter, max_steps: int = 50) -> None:
        self.environment = environment
        self.max_steps = max_steps

    def build_prompt(self, example: TaskExample, agent_spec: AgentSpec, context: dict) -> str:
        admissible = context.get("admissible_actions")
        admissible_text = "\n".join(f"- {action}" for action in admissible) if admissible else "unknown"
        memory_section = self._memory_section(context)
        return (
            f"Goal: {example.input.get('instruction')}\n"
            f"Observation: {context.get('observation', '')}\n"
            f"Admissible actions:\n{admissible_text}\n"
            f"{memory_section}"
            "Return one action."
        )

    def run_episode(self, example, mas, llm, memory) -> Trajectory:
        state = self.environment.reset(example)
        trajectory = Trajectory(
            run_id=make_run_id("trajectory"),
            example_id=example.example_id,
            dataset_name=example.dataset_name,
            mas_type=mas.mas_type,
            model_name=llm.model_name,
            messages=[],
            metadata={"task_type": self.task_type},
        )
        context = {"history": [], "admissible_actions": state.admissible_actions}
        for _ in range(self.max_steps):
            context["observation"] = state.observation
            context["step_id"] = state.step_id
            decision = mas.act(example, state.observation, self, llm, memory, context)
            trajectory.messages.extend(decision.messages)
            next_state = self.environment.step(decision.action)
            step = EnvironmentStep(
                episode_id=next_state.episode_id,
                step_id=next_state.step_id,
                action=decision.action,
                observation=next_state.observation,
                reward=next_state.score,
                done=next_state.done,
                info=next_state.metadata,
            )
            trajectory.environment_steps.append(step)
            context["history"].append(step.model_dump(mode="json"))
            state = next_state
            context["admissible_actions"] = state.admissible_actions
            if state.done:
                break
        trajectory.final_answer = {
            "success": bool(trajectory.environment_steps and trajectory.environment_steps[-1].info.get("success")),
            "steps": len(trajectory.environment_steps),
            "final_observation": state.observation,
        }
        trajectory.usage = self._usage_from_messages(trajectory.messages)
        return trajectory

    def parse_final_answer(self, trajectory: Trajectory):
        return trajectory.final_answer

    def evaluate(self, prediction, example: TaskExample) -> dict:
        if isinstance(prediction, dict):
            return {
                "success": bool(prediction.get("success")),
                "steps": prediction.get("steps", 0),
            }
        return {"success": False, "steps": 0}

    def evaluate_trajectory(self, trajectory: Trajectory) -> dict:
        return compute_interactive_metrics(trajectory)

    def _usage_from_messages(self, messages) -> dict:
        usage: dict[str, float] = {}
        for message in messages:
            message_usage = message.metadata.get("llm_usage", {})
            for key, value in message_usage.items():
                if isinstance(value, (int, float)):
                    usage[key] = usage.get(key, 0) + value
        return usage
