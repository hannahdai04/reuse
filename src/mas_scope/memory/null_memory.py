"""No-op memory provider for Phase 1."""

from __future__ import annotations

from mas_scope.core.registry import registry
from mas_scope.core.types import AgentSpec, TaskExample
from mas_scope.memory.base import MemoryProvider


@registry.register_memory_provider("null")
class NullMemoryProvider(MemoryProvider):
    def __init__(self, **kwargs) -> None:
        self.last_retrieval_metadata: dict = {}

    def retrieve(self, example: TaskExample, agent_spec: AgentSpec, context: dict) -> list[dict]:
        self.last_retrieval_metadata = {
            "candidate_count": 0,
            "selected_ids": [],
            "agent_order": context.get("agent_order", []),
            "agent_mask_rows": [],
            "decisions": [],
            "policy_mode": "null",
            "errors": [],
        }
        return []

    def update(self, example: TaskExample, trajectory, result) -> None:
        return None
