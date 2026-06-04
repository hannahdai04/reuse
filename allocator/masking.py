"""LLM-based memory-agent masking."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from mas_scope.core.types import TaskExample

from allocator.json_utils import parse_json_object


def generate_mask(
    llm: Any,
    example: TaskExample,
    memories: list[dict[str, Any]],
    agents: list[str],
    max_tokens: int = 1200,
) -> dict[str, Any]:
    if not memories or not agents:
        return {"mask_matrix": [], "mask_reasons": [], "errors": []}
    try:
        response = llm.generate(
            [
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": _user_prompt(example, memories, agents)},
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
        payload = parse_json_object(response.content)
        return _validate_payload(payload, memories, agents)
    except Exception as exc:
        return _all_zero(memories, agents, f"masking_error:{exc}")


def _system_prompt() -> str:
    return """You are the Masking module in a four-stage memory allocator:
Retrieval -> Masking -> Selection -> Realization.

Your only job is Masking. Given a target task, ordered agents, and retrieved reusable experience memories, decide whether each memory is allowed for each agent.

Decision meaning:
- z_ij = 1: memory e_i is relevant and useful for agent a_j's role in the target task.
- z_ij = 0: memory e_i should not be shown to agent a_j because it is irrelevant, noisy, redundant for that role, or could mislead the agent.
- agent_scores[j] should be a confidence/usefulness score from 0.0 to 1.0 for agent a_j. Use 0.0 when mask is 0.

Important constraints:
- Do not solve the target task.
- Do not choose the final memory budget; Selection will do that later.
- Do not rewrite memories; Realization will do that later.
- Do not treat producer_agent or producer_role as the future recipient. They are provenance only.
- A memory about team coordination can be useful to multiple agents.
- A memory about critic behavior is usually most useful to critic agents and sometimes summarizers.
- A memory about actor reasoning or decomposition is usually useful to actor agents and sometimes critics.
- A memory about summarization/aggregation is usually useful to summarizer agents.
- If the memory contains source-task answers, entities, or facts, ignore those specifics and judge only the reusable lesson.
- If uncertain, prefer 0 over over-sharing.

Return only valid JSON. Do not include markdown or commentary. Use the exact memory_id values and a binary agent_mask aligned exactly with agent_order."""


def _user_prompt(example: TaskExample, memories: list[dict[str, Any]], agents: list[str]) -> str:
    payload = {
        "target_task": {
            "target_task_id": example.example_id,
            "dataset_name": example.dataset_name,
            "task_type": example.task_type,
            "input": example.input,
        },
        "agent_order": agents,
        "decision_instructions": [
            "Evaluate each memory-agent pair independently.",
            "Use 1 only when the memory can change that agent's behavior in the target MAS run.",
            "Use 0 when the memory is merely a source-task fact, final answer, generic advice, or not relevant to the agent role.",
            "Keep masks sparse enough to avoid context pollution.",
            "Every candidate memory must appear exactly once in decisions.",
            "Each reasons list must have the same length and order as agent_order.",
        ],
        "reason_tag_definitions": {
            "role_match": "The memory directly matches this agent's role responsibility.",
            "coordination": "The memory concerns communication, handoff, aggregation, or team-level behavior.",
            "critic": "The memory concerns critique, verification, error detection, or feedback quality.",
            "retrieval_noise": "The memory was retrieved but does not fit the target task.",
            "context_noise": "The memory may distract or pollute the prompt for most agents.",
            "not_applicable": "The reusable lesson does not apply to this target setting.",
            "other": "Use only if none of the above tags fit.",
        },
        "candidate_memories": [
            {
                "memory_id": memory.get("memory_id"),
                "condition": memory.get("condition"),
                "experience": memory.get("experience"),
                "evidence": memory.get("evidence"),
                "source": memory.get("source"),
                "text": str(memory.get("text") or "")[:900],
            }
            for memory in memories
        ],
        "required_output": {
            "agent_order": agents,
            "decisions": [
                {
                    "memory_id": "memory id",
                    "agent_mask": [0 for _ in agents],
                    "agent_scores": [0.0 for _ in agents],
                    "reasons": ["short per-agent reason aligned with agent_order"] * len(agents),
                    "reason_tag": "role_match|coordination|critic|retrieval_noise|context_noise|not_applicable|other",
                }
            ],
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def _validate_payload(payload: dict[str, Any], memories: list[dict[str, Any]], agents: list[str]) -> dict[str, Any]:
    errors: list[str] = []
    memory_ids = [str(memory.get("memory_id")) for memory in memories]
    known = set(memory_ids)
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        return _all_zero(memories, agents, "invalid_mask_payload")
    if payload.get("agent_order") not in (None, agents):
        errors.append("agent_order_mismatch")

    by_id: dict[str, dict[str, Any]] = {}
    for raw in decisions:
        if not isinstance(raw, dict):
            errors.append("invalid_decision_item")
            continue
        memory_id = str(raw.get("memory_id"))
        if memory_id not in known:
            errors.append(f"unknown_memory_id:{memory_id}")
            continue
        mask = raw.get("agent_mask")
        if not isinstance(mask, list) or len(mask) != len(agents):
            errors.append(f"invalid_mask:{memory_id}")
            continue
        try:
            clean_mask = [1 if int(value) else 0 for value in mask]
        except (TypeError, ValueError):
            errors.append(f"invalid_mask_value:{memory_id}")
            continue
        scores = raw.get("agent_scores")
        if not isinstance(scores, list) or len(scores) != len(agents):
            clean_scores = [1.0 if value else 0.0 for value in clean_mask]
        else:
            clean_scores = []
            for value, mask_value in zip(scores, clean_mask):
                try:
                    score = float(value)
                except (TypeError, ValueError):
                    score = 1.0 if mask_value else 0.0
                if not mask_value:
                    score = 0.0
                clean_scores.append(max(0.0, min(1.0, score)))
        reasons = raw.get("reasons")
        if not isinstance(reasons, list) or len(reasons) != len(agents):
            reason_text = str(raw.get("reason") or raw.get("reason_tag") or "")
            reasons = [reason_text for _ in agents]
        by_id[memory_id] = {
            "memory_id": memory_id,
            "reasons": [str(reason) for reason in reasons],
            "reason_tag": str(raw.get("reason_tag") or "other"),
            "raw_reason": str(raw.get("reason") or ""),
            "agent_mask": clean_mask,
            "agent_scores": clean_scores,
        }

    mask_matrix: list[list[int]] = []
    agent_score_matrix: list[list[float]] = []
    mask_reasons: list[dict[str, Any]] = []
    for memory_id in memory_ids:
        decision = by_id.get(memory_id)
        if decision is None:
            errors.append(f"missing_decision:{memory_id}")
            mask = [0 for _ in agents]
            scores = [0.0 for _ in agents]
            reasons = ["missing decision" for _ in agents]
            tag = "missing_decision"
            raw_reason = ""
        else:
            mask = decision["agent_mask"]
            scores = decision["agent_scores"]
            reasons = decision["reasons"]
            tag = decision["reason_tag"]
            raw_reason = decision["raw_reason"]
        mask_matrix.append(mask)
        agent_score_matrix.append(scores)
        mask_reasons.append(
            {
                "memory_id": memory_id,
                "reasons": reasons,
                "reason_tag": tag,
                "raw_reason": raw_reason,
            }
        )
    return {"mask_matrix": mask_matrix, "agent_score_matrix": agent_score_matrix, "mask_reasons": mask_reasons, "errors": errors}


def _all_zero(memories: list[dict[str, Any]], agents: list[str], error: str) -> dict[str, Any]:
    return {
        "mask_matrix": [[0 for _ in agents] for _ in memories],
        "agent_score_matrix": [[0.0 for _ in agents] for _ in memories],
        "mask_reasons": [
            {
                "memory_id": str(memory.get("memory_id")),
                "reasons": [error for _ in agents],
                "reason_tag": "allocator_error",
                "raw_reason": error,
            }
            for memory in memories
        ],
        "errors": [error],
    }


def reason_tag_counts(mask_reasons: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(item.get("reason_tag") or "unknown") for item in mask_reasons))
