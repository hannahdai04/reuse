"""Deterministic mock ALFWorld environment."""

from __future__ import annotations

from mas_scope.core.types import EnvironmentState, TaskExample
from mas_scope.environments.base import BaseEnvironmentAdapter


class MockAlfworldEnvironment(BaseEnvironmentAdapter):
    valid_actions = ["look", "take apple", "put apple in fridge"]

    def __init__(self, max_steps: int = 5) -> None:
        self.max_steps = max_steps
        self.episode_id = "alfworld_mock_episode"
        self.step_id = 0
        self.done = False
        self.has_apple = False

    def reset(self, example: TaskExample) -> EnvironmentState:
        self.episode_id = example.example_id
        self.step_id = 0
        self.done = False
        self.has_apple = False
        return EnvironmentState(
            episode_id=self.episode_id,
            step_id=self.step_id,
            observation="You are in a kitchen. You see an apple on the table.",
            admissible_actions=self.valid_actions,
            score=0.0,
            done=False,
            metadata={"environment": "alfworld-mock"},
        )

    def step(self, action: str) -> EnvironmentState:
        if self.done:
            return self._state("Episode is already done.", done=True, score=1.0)
        self.step_id += 1
        normalized = action.strip().lower()
        invalid = normalized not in self.valid_actions
        success = False
        if invalid:
            observation = f"Invalid action: {action}"
        elif normalized == "look":
            observation = "You are in a kitchen. You see an apple on the table."
        elif normalized == "take apple":
            self.has_apple = True
            observation = "You take the apple. You are holding the apple."
        else:
            success = True
            observation = "You put the apple in the fridge. Task complete."
        if success:
            self.done = True
        stopped_by_max_steps = self.step_id >= self.max_steps and not self.done
        if stopped_by_max_steps:
            self.done = True
        return self._state(
            observation,
            done=self.done,
            score=1.0 if success else 0.0,
            invalid_action=invalid,
            success=success,
            stopped_by_max_steps=stopped_by_max_steps,
        )

    def close(self) -> None:
        return None

    def get_action_space(self) -> list[str] | None:
        return list(self.valid_actions)

    def _state(self, observation: str, done: bool, score: float, **metadata) -> EnvironmentState:
        return EnvironmentState(
            episode_id=self.episode_id,
            step_id=self.step_id,
            observation=observation,
            admissible_actions=self.valid_actions,
            score=score,
            done=done,
            metadata={"environment": "alfworld-mock", **metadata},
        )
