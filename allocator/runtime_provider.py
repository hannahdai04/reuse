"""Runtime memory provider for allocator ablations."""

from __future__ import annotations

from typing import Any

from mas_scope.core.types import AgentSpec, TaskExample
from mas_scope.memory.base import MemoryProvider

from allocator.masking import generate_mask
from allocator.realization import realize_memories
from allocator.retrieval import retrieve_top_k
from allocator.selection import select_memories


class AllocatorRuntimeMemoryProvider(MemoryProvider):
    """Computes and serves allocator decisions through the existing MAS memory hook."""

    def __init__(
        self,
        setting: str,
        memories: list[dict[str, Any]],
        allocator_llm: Any,
        retrieval_top_k: int = 10,
        per_agent_memory_k: int = 3,
        global_chunk_size: int = 20,
        mask_allowed_per_agent_k: int = 10,
        masking_max_tokens: int = 1200,
        selection_max_tokens: int = 1000,
        realization_max_tokens: int = 1600,
    ) -> None:
        if setting not in {"B1", "B2", "B3", "B4", "G2", "G3"}:
            raise ValueError(f"Unsupported allocator setting: {setting}.")
        self.setting = setting
        self.memories = memories
        self.allocator_llm = allocator_llm
        self.retrieval_top_k = retrieval_top_k
        self.per_agent_memory_k = per_agent_memory_k
        self.global_chunk_size = global_chunk_size
        self.mask_allowed_per_agent_k = mask_allowed_per_agent_k
        self.masking_max_tokens = masking_max_tokens
        self.selection_max_tokens = selection_max_tokens
        self.realization_max_tokens = realization_max_tokens
        self.records: dict[str, dict[str, Any]] = {}
        self.last_retrieval_metadata: dict[str, Any] = {}

    def retrieve(self, example: TaskExample, agent_spec: AgentSpec, context: dict) -> list[dict]:
        agents = list(context.get("agent_order") or [agent_spec.role])
        current_agent = str(context.get("current_agent") or agent_spec.role)
        record = self._ensure_record(example, agents)
        realized_by_agent = record.get("realized_memories") or {}
        selected = [self._injectable_memory(item, record) for item in realized_by_agent.get(current_agent, [])]
        self.last_retrieval_metadata = self._metadata(record, agents, current_agent, selected)
        return selected

    def update(self, example: TaskExample, trajectory, result) -> None:
        return None

    def get_record(self, example_id: str) -> dict[str, Any] | None:
        return self.records.get(example_id)

    def _ensure_record(self, example: TaskExample, agents: list[str]) -> dict[str, Any]:
        record = self.records.get(example.example_id)
        if record is not None:
            return record

        retrieved = self._candidate_memories(example)
        allocator_errors: list[str] = []
        if self.setting == "B1":
            mask_matrix = [[1 for _ in agents] for _ in retrieved]
            mask_reasons = [
                {
                    "memory_id": memory["memory_id"],
                    "reasons": ["shared to all agents" for _ in agents],
                    "reason_tag": "all_shared",
                    "raw_reason": "B1 retrieval_all_shared",
                }
                for memory in retrieved
            ]
            selected_matrix = [list(row) for row in mask_matrix]
            agent_score_matrix = [[1.0 for _ in agents] for _ in retrieved]
            realized = realize_memories(self.allocator_llm, example, retrieved, agents, selected_matrix, use_llm=False)
            cap_stats = _empty_cap_stats(agents)
        else:
            if self.setting in {"G2", "G3"}:
                mask = self._chunked_mask(example, retrieved, agents)
            else:
                mask = generate_mask(
                    self.allocator_llm,
                    example,
                    retrieved,
                    agents,
                    max_tokens=self.masking_max_tokens,
                )
            mask_matrix = mask["mask_matrix"]
            agent_score_matrix = mask.get("agent_score_matrix") or [[1.0 if int(value) else 0.0 for value in row] for row in mask_matrix]
            mask_reasons = mask["mask_reasons"]
            allocator_errors.extend(mask.get("errors") or [])
            cap_result = apply_allowed_cap(
                retrieved,
                agents,
                mask_matrix,
                self.mask_allowed_per_agent_k if self.setting in {"G2", "G3"} else None,
                agent_score_matrix=agent_score_matrix,
            )
            mask_matrix = cap_result["mask_matrix"]
            agent_score_matrix = cap_result["agent_score_matrix"]
            cap_stats = cap_result["cap_stats"]
            if self.setting in {"B2", "G2"}:
                selected_matrix = [list(row) for row in mask_matrix]
            else:
                selection = select_memories(
                    self.allocator_llm,
                    example,
                    retrieved,
                    agents,
                    mask_matrix,
                    self.per_agent_memory_k,
                    use_llm=True,
                    max_tokens=self.selection_max_tokens,
                )
                selected_matrix = selection["selected_matrix"]
                allocator_errors.extend(selection.get("errors") or [])
            realized = realize_memories(
                self.allocator_llm,
                example,
                retrieved,
                agents,
                selected_matrix,
                use_llm=self.setting == "B4",
                max_tokens=self.realization_max_tokens,
            )
            allocator_errors.extend(realized.get("errors") or [])

        record = {
            "target_task_id": example.example_id,
            "setting": self.setting,
            "retrieval_top_k": self.retrieval_top_k,
            "per_agent_memory_k": self.per_agent_memory_k,
            "global_chunk_size": self.global_chunk_size,
            "mask_allowed_per_agent_k": self.mask_allowed_per_agent_k,
            "global_masking": self.setting in {"G2", "G3"},
            "allowed_cap_stats": cap_stats,
            "retrieved_memories": [_public_memory(memory) for memory in retrieved],
            "agents": agents,
            "mask_matrix": mask_matrix,
            "agent_score_matrix": agent_score_matrix,
            "mask_reasons": mask_reasons,
            "selected_matrix": selected_matrix,
            "realized_memories": realized["realized_memories"],
            "task_result": None,
            "allocator_errors": allocator_errors,
        }
        self.records[example.example_id] = record
        return record

    def _candidate_memories(self, example: TaskExample) -> list[dict[str, Any]]:
        if self.setting in {"G2", "G3"}:
            candidates = []
            for rank, memory in enumerate(self.memories, start=1):
                item = dict(memory)
                item.setdefault("retrieval_score", 0.0)
                item["retrieval_rank"] = rank
                candidates.append(item)
            return candidates
        return retrieve_top_k(example, self.memories, self.retrieval_top_k)

    def _chunked_mask(self, example: TaskExample, memories: list[dict[str, Any]], agents: list[str]) -> dict[str, Any]:
        if self.global_chunk_size <= 0:
            chunk_size = len(memories) or 1
        else:
            chunk_size = self.global_chunk_size
        mask_matrix: list[list[int]] = []
        mask_reasons: list[dict[str, Any]] = []
        errors: list[str] = []
        for start in range(0, len(memories), chunk_size):
            chunk = memories[start : start + chunk_size]
            chunk_result = generate_mask(
                self.allocator_llm,
                example,
                chunk,
                agents,
                max_tokens=self.masking_max_tokens,
            )
            mask_matrix.extend(chunk_result.get("mask_matrix") or [])
            mask_reasons.extend(chunk_result.get("mask_reasons") or [])
            errors.extend(f"chunk_{start // chunk_size}:{error}" for error in chunk_result.get("errors") or [])
        return {"mask_matrix": mask_matrix, "mask_reasons": mask_reasons, "errors": errors}

    def _injectable_memory(self, item: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
        memory = {
            "memory_id": item.get("memory_id"),
            "original_memory_id": item.get("original_memory_id"),
            "memory_namespace": item.get("memory_namespace"),
            "text": item.get("text"),
            "source_example_id": item.get("memory_namespace"),
            "decision": {
                "score": item.get("retrieval_score", ""),
                "policy": self.setting,
                "realization_type": item.get("realization_type", "raw"),
            },
        }
        return memory

    def _metadata(
        self,
        record: dict[str, Any],
        agents: list[str],
        current_agent: str,
        selected: list[dict[str, Any]],
    ) -> dict[str, Any]:
        decisions = []
        retrieved = record.get("retrieved_memories") or []
        selected_ids = {str(memory.get("memory_id")) for memory in selected}
        for row, memory in enumerate(retrieved):
            mask = record.get("mask_matrix", [])
            decisions.append(
                {
                    "memory_id": memory.get("memory_id"),
                    "score": memory.get("retrieval_score", 0.0),
                    "agent_mask": mask[row] if row < len(mask) else [],
                    "policy": self.setting,
                    "selected_for_current_agent": str(memory.get("memory_id")) in selected_ids,
                }
            )
        return {
            "candidate_count": len(retrieved),
            "selected_ids": list(selected_ids),
            "agent_order": agents,
            "current_agent": current_agent,
            "agent_mask_rows": record.get("mask_matrix") or [],
            "decisions": decisions,
            "policy_mode": self.setting,
            "errors": record.get("allocator_errors") or [],
        }


def _public_memory(memory: dict[str, Any]) -> dict[str, Any]:
    return {
        "memory_id": memory.get("memory_id"),
        "original_memory_id": memory.get("original_memory_id"),
        "memory_namespace": memory.get("memory_namespace"),
        "retrieval_rank": memory.get("retrieval_rank"),
        "retrieval_score": memory.get("retrieval_score"),
        "source": memory.get("source"),
        "source_task_description": memory.get("source_task_description"),
        "condition": memory.get("condition"),
        "experience": memory.get("experience"),
        "evidence": memory.get("evidence"),
        "text": memory.get("text"),
    }


def apply_allowed_cap(
    memories: list[dict[str, Any]],
    agents: list[str],
    mask_matrix: list[list[int]],
    cap: int | None,
    agent_score_matrix: list[list[float]] | None = None,
) -> dict[str, Any]:
    before_counts = _allowed_counts(mask_matrix, agents)
    scores = agent_score_matrix or [[1.0 if int(value) else 0.0 for value in row] for row in mask_matrix]
    if cap is None or cap < 0:
        return {
            "mask_matrix": [list(row) for row in mask_matrix],
            "agent_score_matrix": [list(row) for row in scores],
            "cap_stats": {
                "enabled": False,
                "cap": cap,
                "allowed_before_by_agent": dict(zip(agents, before_counts)),
                "allowed_after_by_agent": dict(zip(agents, before_counts)),
                "dropped_by_agent": {agent: 0 for agent in agents},
                "dropped_count": 0,
                "dropped_ratio": 0.0,
                "avg_allowed_before_cap": _avg(before_counts),
                "avg_allowed_after_cap": _avg(before_counts),
            },
        }

    capped = [list(row) for row in mask_matrix]
    capped_scores = [list(row) for row in scores]
    dropped_by_agent = {agent: 0 for agent in agents}
    for col, agent in enumerate(agents):
        allowed_rows = [
            row
            for row, mask in enumerate(capped)
            if col < len(mask) and int(mask[col]) == 1
        ]
        allowed_rows.sort(
            key=lambda row: (
                -float(scores[row][col]) if col < len(scores[row]) else 0.0,
                -float(memories[row].get("retrieval_score") or 0.0),
                int(memories[row].get("retrieval_rank") or 999999),
                str(memories[row].get("memory_id")),
            )
        )
        keep = set(allowed_rows[:cap])
        for row in allowed_rows[cap:]:
            capped[row][col] = 0
            if col < len(capped_scores[row]):
                capped_scores[row][col] = 0.0
            dropped_by_agent[agent] += 1

    after_counts = _allowed_counts(capped, agents)
    dropped_count = sum(dropped_by_agent.values())
    before_total = sum(before_counts)
    return {
        "mask_matrix": capped,
        "agent_score_matrix": capped_scores,
        "cap_stats": {
            "enabled": True,
            "cap": cap,
            "allowed_before_by_agent": dict(zip(agents, before_counts)),
            "allowed_after_by_agent": dict(zip(agents, after_counts)),
            "dropped_by_agent": dropped_by_agent,
            "dropped_count": dropped_count,
            "dropped_ratio": dropped_count / before_total if before_total else 0.0,
            "avg_allowed_before_cap": _avg(before_counts),
            "avg_allowed_after_cap": _avg(after_counts),
        },
    }


def _empty_cap_stats(agents: list[str]) -> dict[str, Any]:
    return {
        "enabled": False,
        "cap": None,
        "allowed_before_by_agent": {agent: 0 for agent in agents},
        "allowed_after_by_agent": {agent: 0 for agent in agents},
        "dropped_by_agent": {agent: 0 for agent in agents},
        "dropped_count": 0,
        "dropped_ratio": 0.0,
        "avg_allowed_before_cap": 0.0,
        "avg_allowed_after_cap": 0.0,
    }


def _allowed_counts(mask_matrix: list[list[int]], agents: list[str]) -> list[int]:
    counts = [0 for _ in agents]
    for row in mask_matrix:
        for col, value in enumerate(row):
            if col < len(counts):
                counts[col] += 1 if int(value) else 0
    return counts


def _avg(values: list[int] | list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
