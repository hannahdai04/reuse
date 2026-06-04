"""Optional real ALFWorld adapter."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from mas_scope.core.exceptions import ConfigurationError, EnvironmentDependencyError
from mas_scope.core.types import EnvironmentState, TaskExample
from mas_scope.environments.base import BaseEnvironmentAdapter


class AlfworldEnvironment(BaseEnvironmentAdapter):
    """Thin adapter over ALFWorld's TextWorld-style environment API."""

    def __init__(
        self,
        config: dict | None = None,
        config_path: str | Path | None = None,
        env_type: str | None = None,
        train_eval: str | None = None,
        batch_size: int = 1,
    ) -> None:
        try:
            from alfworld.agents.environment import get_environment
            import alfworld.agents.modules.generic as generic
        except ImportError as exc:
            raise EnvironmentDependencyError(
                "ALFWorld is not installed. Install ALFWorld separately to use the real adapter."
            ) from exc
        self._get_environment = get_environment
        self._generic = generic
        self.config = config or self._load_config(config_path)
        self.env_type = env_type or self.config.get("env", {}).get("type")
        if not self.env_type:
            raise ConfigurationError("ALFWorld config must define env.type or pass env_type.")
        self.train_eval = train_eval
        self.batch_size = batch_size
        self._env = None
        self.episode_id = "alfworld_episode"
        self.step_id = 0
        self._last_info: dict[str, Any] = {}
        self._last_admissible_actions: list[str] | None = None

    def reset(self, example: TaskExample) -> EnvironmentState:
        self.episode_id = example.example_id
        self.step_id = 0
        self._ensure_env(example)
        obs, info = self._env.reset()
        observation = self._first(obs)
        info = self._first_info(info)
        self._last_info = info
        self._last_admissible_actions = self._admissible_from_info(info)
        return EnvironmentState(
            episode_id=self.episode_id,
            step_id=self.step_id,
            observation=observation,
            admissible_actions=self._last_admissible_actions,
            score=self._score_from_info(info),
            done=False,
            metadata={"environment": "alfworld", "info": info},
        )

    def step(self, action: str) -> EnvironmentState:
        if self._env is None:
            raise RuntimeError("ALFWorld environment must be reset before step().")
        self.step_id += 1
        obs, scores, dones, infos = self._env.step([action])
        observation = self._first(obs)
        done = bool(self._first(dones))
        info = self._first_info(infos)
        score = self._first(scores)
        self._last_info = info
        self._last_admissible_actions = self._admissible_from_info(info)
        success = bool(info.get("won") or info.get("success") or (done and score and float(score) > 0))
        return EnvironmentState(
            episode_id=self.episode_id,
            step_id=self.step_id,
            observation=observation,
            admissible_actions=self._last_admissible_actions,
            score=float(score) if isinstance(score, (int, float)) else self._score_from_info(info),
            done=done,
            metadata={"environment": "alfworld", "info": info, "success": success},
        )

    def close(self) -> None:
        if self._env is not None and hasattr(self._env, "close"):
            self._env.close()
        self._env = None

    def get_action_space(self) -> list[str] | None:
        return list(self._last_admissible_actions) if self._last_admissible_actions is not None else None

    def _load_config(self, config_path: str | Path | None) -> dict:
        configured_path = config_path or os.getenv("ALFWORLD_CONFIG")
        if configured_path:
            path = Path(configured_path)
            if not path.exists():
                raise ConfigurationError(f"ALFWorld config not found: {path}")
            with path.open("r", encoding="utf-8") as handle:
                return yaml.safe_load(handle)
        try:
            return self._generic.load_config()
        except BaseException as exc:
            raise ConfigurationError(
                "ALFWorld config is required. Pass environment.config.config_path or set ALFWORLD_CONFIG."
            ) from exc

    def _ensure_env(self, example: TaskExample) -> None:
        if self._env is not None:
            return
        train_eval = self.train_eval or self._split_to_train_eval(example.split)
        env_cls = self._get_environment(self.env_type)
        self._env = env_cls(self.config, train_eval=train_eval).init_env(batch_size=self.batch_size)

    def _split_to_train_eval(self, split: str) -> str:
        if split == "train":
            return "train"
        if split in {"eval_in_distribution", "eval_out_of_distribution"}:
            return split
        if split in {"valid_seen", "valid_train", "dev", "validation"}:
            return "eval_in_distribution"
        if split in {"valid_unseen", "test"}:
            return "eval_out_of_distribution"
        return "eval_out_of_distribution"

    def _admissible_from_info(self, info: dict) -> list[str] | None:
        commands = info.get("admissible_commands") or info.get("valid")
        commands = self._first(commands)
        return list(commands) if commands is not None else None

    def _score_from_info(self, info: dict) -> float | None:
        score = info.get("score") or info.get("scores")
        score = self._first(score)
        return float(score) if isinstance(score, (int, float)) else None

    def _first(self, value):
        if isinstance(value, (list, tuple)):
            return value[0] if value else None
        return value

    def _first_info(self, value) -> dict:
        first = self._first(value)
        return first if isinstance(first, dict) else {}
