"""Base interface for interactive environments."""

from __future__ import annotations

from abc import ABC, abstractmethod

from mas_scope.core.types import EnvironmentState, TaskExample


class BaseEnvironmentAdapter(ABC):
    @abstractmethod
    def reset(self, example: TaskExample) -> EnvironmentState:
        ...

    @abstractmethod
    def step(self, action: str) -> EnvironmentState:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    @abstractmethod
    def get_action_space(self) -> list[str] | None:
        ...
