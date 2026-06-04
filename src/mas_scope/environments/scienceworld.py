"""Optional real ScienceWorld adapter."""

from __future__ import annotations

from typing import Any

from mas_scope.core.exceptions import ConfigurationError, EnvironmentDependencyError
from mas_scope.core.types import EnvironmentState, TaskExample
from mas_scope.environments.base import BaseEnvironmentAdapter


class ScienceWorldEnvironment(BaseEnvironmentAdapter):
    """Thin adapter over ScienceWorldEnv."""

    def __init__(
        self,
        env_step_limit: int = 100,
        simplification: str | None = None,
        generate_gold_path: bool = False,
        jar_path: str | None = None,
        server_path: str | None = None,
    ) -> None:
        try:
            try:
                from scienceworld import ScienceWorldEnv
            except ImportError:
                from scienceworld.scienceworld import ScienceWorldEnv
        except ImportError as exc:
            raise EnvironmentDependencyError(
                "ScienceWorld is not installed. Install ScienceWorld separately to use the real adapter."
            ) from exc
        jar_or_server_path = jar_path or server_path
        self._env = ScienceWorldEnv("", jar_or_server_path, envStepLimit=env_step_limit)
        self.env_step_limit = env_step_limit
        self.simplification = simplification
        self.generate_gold_path = generate_gold_path
        self.episode_id = "scienceworld_episode"
        self.step_id = 0
        self._last_info: dict[str, Any] = {}
        self._last_admissible_actions: list[str] | None = None

    def reset(self, example: TaskExample) -> EnvironmentState:
        self.episode_id = example.example_id
        self.step_id = 0
        task_name = example.input.get("task_name")
        if task_name in (None, ""):
            task_name = example.input.get("task_id")
        if task_name in (None, ""):
            raise ConfigurationError("ScienceWorld examples require input.task_name or input.task_id.")
        variation_id = int(example.input.get("variation_id") or 0)
        simplification = (
            example.metadata.get("simplification")
            or example.input.get("simplification")
            or self.simplification
            or ""
        )
        self._env.load(task_name, variation_id, simplification, self.generate_gold_path)
        observation, info = self._env.reset()
        self._last_info = info if isinstance(info, dict) else {}
        self._last_admissible_actions = self._admissible_from_info(self._last_info)
        return EnvironmentState(
            episode_id=self.episode_id,
            step_id=self.step_id,
            observation=observation,
            admissible_actions=self._last_admissible_actions,
            score=self._score_from_info(self._last_info),
            done=False,
            metadata={"environment": "scienceworld", "info": self._last_info},
        )

    def step(self, action: str) -> EnvironmentState:
        self.step_id += 1
        observation, reward, done, info = self._env.step(action)
        info = info if isinstance(info, dict) else {}
        self._last_info = info
        self._last_admissible_actions = self._admissible_from_info(info)
        score = self._score_from_info(info)
        success = bool(done and (score is not None and score >= 100))
        return EnvironmentState(
            episode_id=self.episode_id,
            step_id=self.step_id,
            observation=observation,
            admissible_actions=self._last_admissible_actions,
            score=score if score is not None else float(reward) if isinstance(reward, (int, float)) else None,
            done=bool(done),
            metadata={"environment": "scienceworld", "info": info, "success": success, "reward": reward},
        )

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()
        elif hasattr(self._env, "shutdown"):
            self._env.shutdown()

    def get_action_space(self) -> list[str] | None:
        return list(self._last_admissible_actions) if self._last_admissible_actions is not None else None

    def _admissible_from_info(self, info: dict) -> list[str] | None:
        actions = info.get("valid")
        if actions is None and hasattr(self._env, "get_valid_action_object_combinations"):
            actions = self._env.get_valid_action_object_combinations()
        if actions is None and hasattr(self._env, "getValidActionObjectCombinations"):
            actions = self._env.getValidActionObjectCombinations()
        if actions is None and hasattr(self._env, "get_valid_action_object_combinations_with_templates"):
            actions = self._env.get_valid_action_object_combinations_with_templates()
        return self._normalize_actions(actions)

    def _score_from_info(self, info: dict) -> float | None:
        score = info.get("score")
        return float(score) if isinstance(score, (int, float)) else None

    def _normalize_actions(self, actions) -> list[str] | None:
        if actions is None:
            return None
        normalized = []
        for action in actions:
            if isinstance(action, dict):
                value = action.get("action")
            else:
                value = action
            if value is not None:
                normalized.append(str(value))
        return normalized
