"""Memory selection under per-agent budget."""

from __future__ import annotations

import json
from typing import Any

from mas_scope.core.types import TaskExample

from allocator.json_utils import parse_json_object


def select_memories(
    llm: Any,
    example: TaskExample,
    memories: list[dict[str, Any]],
    agents: list[str],
    mask_matrix: list[list[int]],
    per_agent_k: int,
    use_llm: bool = True,
    max_tokens: int = 1000,
) -> dict[str, Any]:
    if per_agent_k <= 0 or not memories or not agents:
        return {"selected_matrix": [[0 for _ in agents] for _ in memories], "errors": []}
    if not use_llm:
        return {"selected_matrix": _fallback_select(memories, agents, mask_matrix, per_agent_k), "errors": []}

    try:
        response = llm.generate(
            [
                {"role": "system", "content": _system_prompt(per_agent_k)},
                {"role": "user", "content": _user_prompt(example, memories, agents, mask_matrix, per_agent_k)},
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
        payload = parse_json_object(response.content)
        return _validate_payload(payload, memories, agents, mask_matrix, per_agent_k)
    except Exception as exc:
        return {
            "selected_matrix": _fallback_select(memories, agents, mask_matrix, per_agent_k),
            "errors": [f"selection_error:{exc}"],
        }


def _system_prompt(per_agent_k: int) -> str:
    return f"""You are the Selection module in a four-stage memory allocator:
Retrieval -> Masking -> Selection -> Realization.

Your only job is Selection. Masking has already decided which memories are allowed for each agent. For each agent, select at most {per_agent_k} memories from that agent's allowed list.

Selection goal:
- Keep the most useful, least redundant memories for each agent.
- Reduce context pollution.
- Prefer memories that address likely failure modes of the agent's role on the target task.

Decision rules:
- Never select a memory that is not in allowed_memory_ids_by_agent for that agent.
- Select fewer than {per_agent_k} memories if the allowed memories are weak, duplicate, or noisy.
- Prefer specific operational advice over generic advice.
- Prefer memories that change behavior: decomposition, verification, critique, aggregation, tool-use warnings, or failure correction.
- Avoid selecting multiple memories that say the same thing.
- Do not solve the target task.
- Do not rewrite memory text; Realization will do that later.
- Do not select memories because their source task has the same answer or entity.

Return only valid JSON. Do not include markdown or commentary. Use exact agent names and memory_id values."""


def _user_prompt(
    example: TaskExample,
    memories: list[dict[str, Any]],
    agents: list[str],
    mask_matrix: list[list[int]],
    per_agent_k: int,
) -> str:
    allowed = {
        agent: [
            memories[row].get("memory_id")
            for row in range(len(memories))
            if row < len(mask_matrix) and col < len(mask_matrix[row]) and int(mask_matrix[row][col]) == 1
        ]
        for col, agent in enumerate(agents)
    }
    payload = {
        "target_task": {
            "target_task_id": example.example_id,
            "dataset_name": example.dataset_name,
            "task_type": example.task_type,
            "input": example.input,
        },
        "agent_order": agents,
        "per_agent_memory_k": per_agent_k,
        "selection_instructions": [
            "For every agent in agent_order, output one selection object.",
            "memory_ids must be a subset of allowed_memory_ids_by_agent[agent].",
            "memory_ids length must be less than or equal to per_agent_memory_k.",
            "It is valid to output an empty memory_ids list for an agent.",
            "Rank the selected memory_ids from most useful to least useful.",
        ],
        "candidate_memories": [
            {
                "memory_id": memory.get("memory_id"),
                "retrieval_rank": memory.get("retrieval_rank"),
                "retrieval_score": memory.get("retrieval_score"),
                "condition": memory.get("condition"),
                "experience": memory.get("experience"),
                "evidence": memory.get("evidence"),
                "text": str(memory.get("text") or "")[:800],
            }
            for memory in memories
        ],
        "allowed_memory_ids_by_agent": allowed,
        "required_output": {
            "selections": [
                {
                    "agent": agent,
                    "memory_ids": [],
                    "reason": "short reason",
                }
                for agent in agents
            ]
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def _validate_payload(
    payload: dict[str, Any],
    memories: list[dict[str, Any]],
    agents: list[str],
    mask_matrix: list[list[int]],
    per_agent_k: int,
) -> dict[str, Any]:
    errors: list[str] = []
    selected = [[0 for _ in agents] for _ in memories]
    memory_index = {str(memory.get("memory_id")): idx for idx, memory in enumerate(memories)}
    if payload.get("agent_order") not in (None, agents):
        errors.append("agent_order_mismatch")

    selections = payload.get("selections")
    if not isinstance(selections, list):
        return {
            "selected_matrix": _fallback_select(memories, agents, mask_matrix, per_agent_k),
            "errors": ["invalid_selection_payload"],
        }

    for raw in selections:
        if not isinstance(raw, dict):
            errors.append("invalid_selection_item")
            continue
        agent = str(raw.get("agent"))
        if agent not in agents:
            errors.append(f"unknown_agent:{agent}")
            continue
        col = agents.index(agent)
        memory_ids = raw.get("memory_ids")
        if not isinstance(memory_ids, list):
            errors.append(f"invalid_memory_ids:{agent}")
            continue
        count = 0
        for memory_id_raw in memory_ids:
            memory_id = str(memory_id_raw)
            row = memory_index.get(memory_id)
            if row is None:
                errors.append(f"unknown_memory_id:{memory_id}")
                continue
            if row >= len(mask_matrix) or col >= len(mask_matrix[row]) or int(mask_matrix[row][col]) != 1:
                errors.append(f"not_allowed:{agent}:{memory_id}")
                continue
            if count >= per_agent_k:
                errors.append(f"over_budget:{agent}")
                break
            selected[row][col] = 1
            count += 1

    # If an agent got no valid LLM selections but has allowed memories, use deterministic fallback for that agent.
    fallback = _fallback_select(memories, agents, mask_matrix, per_agent_k)
    for col, _agent in enumerate(agents):
        if sum(selected[row][col] for row in range(len(memories))) == 0:
            for row in range(len(memories)):
                selected[row][col] = fallback[row][col]
    return {"selected_matrix": selected, "errors": errors}


def _fallback_select(
    memories: list[dict[str, Any]],
    agents: list[str],
    mask_matrix: list[list[int]],
    per_agent_k: int,
) -> list[list[int]]:
    selected = [[0 for _ in agents] for _ in memories]
    for col, _agent in enumerate(agents):
        allowed_rows = [
            row
            for row, mask in enumerate(mask_matrix)
            if col < len(mask) and int(mask[col]) == 1
        ]
        allowed_rows.sort(
            key=lambda row: (
                int(memories[row].get("retrieval_rank") or 999999),
                -float(memories[row].get("retrieval_score") or 0.0),
                str(memories[row].get("memory_id")),
            )
        )
        for row in allowed_rows[:per_agent_k]:
            selected[row][col] = 1
    return selected
