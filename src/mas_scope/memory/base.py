"""Memory provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from mas_scope.core.types import AgentSpec, TaskExample


class MemoryProvider(ABC):
    @abstractmethod
    def retrieve(self, example: TaskExample, agent_spec: AgentSpec, context: dict) -> list[dict]:
        ...

    @abstractmethod
    def update(self, example: TaskExample, trajectory, result) -> None:
        ...
