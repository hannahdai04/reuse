"""Memory realization for final prompt injection."""

from __future__ import annotations

import json
from typing import Any

from mas_scope.core.types import TaskExample

from allocator.json_utils import parse_json_object


REALIZATION_TYPES = {"raw", "abstract", "warning", "agent_specific"}


def realize_memories(
    llm: Any,
    example: TaskExample,
    memories: list[dict[str, Any]],
    agents: list[str],
    selected_matrix: list[list[int]],
    use_llm: bool = True,
    max_tokens: int = 1600,
) -> dict[str, Any]:
    selected_pairs = _selected_pairs(memories, agents, selected_matrix)
    if not selected_pairs:
        return {"realized_memories": {agent: [] for agent in agents}, "errors": []}
    if not use_llm:
        return {"realized_memories": _raw_realizations(selected_pairs, agents), "errors": []}

    try:
        response = llm.generate(
            [
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": _user_prompt(example, selected_pairs, agents)},
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
        payload = parse_json_object(response.content)
        return _validate_payload(payload, selected_pairs, agents)
    except Exception as exc:
        return {
            "realized_memories": _raw_realizations(selected_pairs, agents),
            "errors": [f"realization_error:{exc}"],
        }


def _system_prompt() -> str:
    return """You are the Realization module in a four-stage memory allocator:
Retrieval -> Masking -> Selection -> Realization.

Your only job is Realization. Selection has already chosen which memories each agent will receive. Rewrite each selected memory into the final text that will be injected into that agent's prompt.

Realization goal:
- Make the memory actionable for the receiving agent's role.
- Preserve the reusable lesson, strategy, warning, or heuristic.
- Preserve concrete checks, decision criteria, failure warnings, and answer-format constraints from the original memory.
- Remove irrelevant source-task details without removing the actionable mechanism.
- Avoid context pollution, but do not over-compress into generic advice.

Allowed realization_type values:
- raw: keep the original reusable memory nearly unchanged when it is already actionable and transferable.
- abstract: generalize source-specific details while preserving the concrete action/check.
- warning: phrase a failure mode as a specific risk to check before answering.
- agent_specific: tailor the advice to the receiving agent's role without losing the original operational detail.

Hard constraints:
- Do not solve the target task.
- Do not add new facts about the target task.
- Do not copy source-task final answers, entity names, dates, or task-specific facts unless they are necessary to understand the reusable lesson.
- Do not mention that this came from a previous trajectory unless that is needed for evidence.
- Do not output long summaries; each realized memory should usually be 1-2 short sentences or a compact checklist sentence.
- Do not replace a concrete memory with vague advice like "verify carefully" or "use evidence"; include what to verify or how to use evidence.
- If rewriting would weaken the memory, use realization_type raw and keep the original reusable text.
- Keep the output useful even if the target task answer is unknown.

Return only valid JSON. Do not include markdown or commentary. Use exact agent names and memory_id values."""


def _user_prompt(example: TaskExample, selected_pairs: list[dict[str, Any]], agents: list[str]) -> str:
    payload = {
        "target_task": {
            "target_task_id": example.example_id,
            "dataset_name": example.dataset_name,
            "task_type": example.task_type,
            "input": example.input,
        },
        "agent_order": agents,
        "realization_instructions": [
            "Create exactly one realization for each selected_memory_agent_pair.",
            "The text field is what will be injected into the agent prompt.",
            "Keep the realized text as an operating rule/checklist item that the agent can apply before answering.",
            "Preserve concrete checks, comparison direction, evidence-grounding steps, answer granularity, and format warnings when present.",
            "Use raw when the source text is already actionable; do not rewrite just to make it shorter.",
            "Use agent_specific only when the receiving role changes the action the agent should take.",
            "Use warning when the memory mainly describes a specific failure mode to avoid.",
            "Use abstract only to remove source-specific details while keeping the original action/check.",
            "Avoid generic rewrites such as 'be careful', 'verify the answer', or 'use evidence' without a concrete check.",
        ],
        "selected_memory_agent_pairs": [
            {
                "agent": pair["agent"],
                "memory_id": pair["memory"]["memory_id"],
                "condition": pair["memory"].get("condition"),
                "experience": pair["memory"].get("experience"),
                "evidence": pair["memory"].get("evidence"),
                "source": pair["memory"].get("source"),
                "text": str(pair["memory"].get("text") or "")[:900],
            }
            for pair in selected_pairs
        ],
        "required_output": {
            "realizations": [
                {
                    "agent": "agent role",
                    "memory_id": "memory id",
                    "realization_type": "raw|abstract|warning|agent_specific",
                    "text": "final memory text to inject",
                }
            ]
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def _validate_payload(
    payload: dict[str, Any],
    selected_pairs: list[dict[str, Any]],
    agents: list[str],
) -> dict[str, Any]:
    errors: list[str] = []
    realized = _raw_realizations(selected_pairs, agents)
    pair_keys = {(pair["agent"], str(pair["memory"]["memory_id"])) for pair in selected_pairs}
    realizations = payload.get("realizations")
    if not isinstance(realizations, list):
        return {"realized_memories": realized, "errors": ["invalid_realization_payload"]}

    by_agent = {agent: [] for agent in agents}
    raw_by_key = {
        (agent, str(item["memory_id"])): item
        for agent, items in realized.items()
        for item in items
    }
    for raw in realizations:
        if not isinstance(raw, dict):
            errors.append("invalid_realization_item")
            continue
        agent = str(raw.get("agent"))
        memory_id = str(raw.get("memory_id"))
        key = (agent, memory_id)
        if key not in pair_keys:
            errors.append(f"unknown_pair:{agent}:{memory_id}")
            continue
        text = str(raw.get("text") or "").strip()
        if not text:
            errors.append(f"empty_realization:{agent}:{memory_id}")
            item = raw_by_key[key]
        else:
            realization_type = str(raw.get("realization_type") or "raw")
            if realization_type not in REALIZATION_TYPES:
                realization_type = "raw"
            item = dict(raw_by_key[key])
            item["text"] = text
            item["realization_type"] = realization_type
        by_agent[agent].append(item)

    for key, item in raw_by_key.items():
        agent, _memory_id = key
        if not any(existing["memory_id"] == item["memory_id"] for existing in by_agent[agent]):
            by_agent[agent].append(item)
    return {"realized_memories": by_agent, "errors": errors}


def _selected_pairs(
    memories: list[dict[str, Any]],
    agents: list[str],
    selected_matrix: list[list[int]],
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for row, memory in enumerate(memories):
        if row >= len(selected_matrix):
            continue
        for col, agent in enumerate(agents):
            if col < len(selected_matrix[row]) and int(selected_matrix[row][col]) == 1:
                pairs.append({"agent": agent, "memory": memory})
    return pairs


def _raw_realizations(selected_pairs: list[dict[str, Any]], agents: list[str]) -> dict[str, list[dict[str, Any]]]:
    realized = {agent: [] for agent in agents}
    for pair in selected_pairs:
        memory = pair["memory"]
        item = {
            "memory_id": str(memory.get("memory_id")),
            "original_memory_id": memory.get("original_memory_id"),
            "memory_namespace": memory.get("memory_namespace"),
            "realization_type": "raw",
            "text": str(memory.get("text") or memory.get("experience") or ""),
            "retrieval_score": memory.get("retrieval_score"),
            "retrieval_rank": memory.get("retrieval_rank"),
        }
        realized[pair["agent"]].append(item)
    return realized
