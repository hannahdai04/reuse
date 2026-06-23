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

Your only job is Selection. Masking has already produced each agent's allowed memory list. Select a compact but sufficient set of reusable experiences for final prompt injection.

Selection goal:
- Keep at most {per_agent_k} memories per agent, but preserve enough useful guidance for the agent to actually change behavior.
- Masking already filtered obvious noise. Selection should mainly remove redundancy and fit the memory budget, not reject most allowed memories again.
- Maximize behavior change: choose memories that tell the agent what to check, avoid, compare, preserve, or verify.
- Minimize context pollution by removing duplicates, source-fact-only memories, and weak role fits.

Decision rules:
- Never select a memory that is not in allowed_memory_ids_by_agent for that agent.
- Do not solve the target task.
- Do not rewrite memory text; Realization will do that later.
- Do not select memories because their source task has the same answer or entity.
- If an agent has multiple allowed memories with distinct process value, select 2 to {per_agent_k} of them rather than collapsing to a single memory.
- Output an empty list only when no allowed memory gives that agent any useful operating rule.
- Prefer memories with a clear condition, actionable experience, and evidence-backed failure or success pattern.
- Avoid selecting multiple memories that say the same thing; keep the most concrete one.
- Prefer a diverse set: one evidence/reasoning rule, one verification or error-prevention rule, and one answer-format or aggregation rule when available.

Role-specific ranking:
- Actor agents: prefer decomposition, evidence grounding, entity disambiguation, comparison, number/date extraction, and avoiding unsupported yes/no shortcuts.
- Critic agents: prefer verification checks, contradiction detection, answer granularity checks, and correcting unsupported or malformed candidates.
- Summarizer agent: prefer candidate arbitration, exact answer preservation, partial-answer repair, and final-format discipline.

Return only valid compact JSON. Do not include markdown, commentary, reasons, or extra fields. Use exact agent names and memory_id values."""


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
            "Do not output an empty list if the agent has allowed memories with usable process guidance.",
            "Rank selected memory_ids from most useful to least useful.",
            "Select only memories that can change this agent's behavior on the target task.",
            "When two or more allowed memories are useful and non-duplicate, select 2 to per_agent_memory_k memories.",
            "Do not select source-specific facts, duplicate lessons, or memories that mainly restate the task format.",
            "Prefer diversity across reasoning, evidence grounding, verification, aggregation, and final-format control.",
        ],
        "candidate_memories": [
            {
                "memory_id": memory.get("memory_id"),
                "retrieval_rank": memory.get("retrieval_rank"),
                "retrieval_score": memory.get("retrieval_score"),
                "condition": str(memory.get("condition") or "")[:260],
                "experience": str(memory.get("experience") or "")[:380],
                "evidence": str(memory.get("evidence") or "")[:220],
            }
            for memory in memories
        ],
        "allowed_memory_ids_by_agent": allowed,
        "required_output": {
            "selections": [
                {
                    "agent": agent,
                    "memory_ids": [],
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
