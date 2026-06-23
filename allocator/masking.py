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

Your only job is Masking. Given a target task, ordered agents, and reusable experience memories, decide which agents can use each memory as an operating rule or checklist item.

Decision meaning:
- z_ij = 1: memory e_i can plausibly improve agent a_j's process on this target task.
- z_ij = 0: memory e_i is clearly irrelevant, source-fact-only, answer-specific, unsupported, or likely to mislead agent a_j.

Important constraints:
- Do not solve the target task.
- Do not choose the final memory budget; Selection will do that later.
- Do not rewrite memories; Realization will do that later.
- Do not treat producer_agent or producer_role as the future recipient. They are provenance only.

Quality gate before role assignment:
- Memories are process guidance, not target facts. They may still be useful even when they are not about the same entities as the target task.
- Use 1 for memories that give a reusable action, warning, verification check, decomposition habit, evidence-use habit, or answer-format habit that an agent can apply.
- Reject only when the memory is clearly not transferable, mostly copies source facts or answers, contradicts the target task, is unsupported by its evidence, or would distract the agent.
- Do not require exact dataset, entity, or question overlap. A memory can be useful through reasoning pattern, evidence handling, verification, or final-answer discipline.
- Avoid both extremes: do not broadcast every memory to every agent, but also do not return all-zero masks unless every memory is clearly unusable.
- For a typical top-k list, route the best few useful memories to the most relevant one or two agents.

Role guidance:
- Actor agents need memories about decomposition, evidence lookup, entity disambiguation, comparison, numerical extraction, bridge reasoning, and avoiding premature yes/no answers.
- Critic agents need memories about checking whether an answer is supported by the target context, catching entity/type/date/number mismatches, and correcting malformed or underspecified answers.
- The summarizer agent needs memories about choosing between candidates, preserving exact answer granularity, handling partial answers, and enforcing the final answer format.
- Team coordination memories may be useful to several agents only when they directly prevent a likely handoff or aggregation failure.

Return only valid compact JSON. Do not include markdown, commentary, reasons, scores, or extra fields.
For each memory, output only memory_id and mask. The mask is a string of 0/1 characters aligned exactly with agent_order.
Example: {"agent_order":["actor","critic"],"decisions":[{"memory_id":"m1","mask":"10"}]}"""


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
            "Treat each memory as an operating rule/checklist item, not as target-task evidence.",
            "Assign 1 when an agent can plausibly use the memory to improve decomposition, evidence use, verification, aggregation, or answer formatting.",
            "Assign 0 when the memory is clearly irrelevant to the agent role, source-fact-only, answer-specific, unsupported, or likely to mislead.",
            "For HotpotQA, prefer memories that improve evidence grounding, entity disambiguation, bridge/comparison reasoning, answer granularity, or exact final format.",
            "For StrategyQA, prefer memories that improve decomposition, yes/no consistency, fact-to-conclusion operations, and contradiction checks.",
            "Use role-specific masks: usually route a useful memory to one or two suitable agents, not all agents.",
            "Do not output all-zero masks for the whole list unless all candidate memories are clearly unusable.",
            "Every candidate memory must appear exactly once in decisions.",
        ],
        "candidate_memories": [
            {
                "memory_id": memory.get("memory_id"),
                "condition": str(memory.get("condition") or "")[:260],
                "experience": str(memory.get("experience") or "")[:360],
                "evidence": str(memory.get("evidence") or "")[:220],
            }
            for memory in memories
        ],
        "required_output": {
            "agent_order": agents,
            "decisions": [
                {
                    "memory_id": "memory id",
                    "mask": "0" * len(agents),
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
        mask = _parse_mask(raw, agents)
        if mask is None:
            errors.append(f"invalid_mask:{memory_id}")
            continue
        clean_mask = mask
        clean_scores = [1.0 if value else 0.0 for value in clean_mask]
        by_id[memory_id] = {
            "memory_id": memory_id,
            "reason_tag": _reason_tag(clean_mask),
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
            tag = "missing_decision"
        else:
            mask = decision["agent_mask"]
            scores = decision["agent_scores"]
            tag = decision["reason_tag"]
        mask_matrix.append(mask)
        agent_score_matrix.append(scores)
        mask_reasons.append(
            {
                "memory_id": memory_id,
                "reason_tag": tag,
            }
        )
    return {"mask_matrix": mask_matrix, "agent_score_matrix": agent_score_matrix, "mask_reasons": mask_reasons, "errors": errors}


def _parse_mask(raw: dict[str, Any], agents: list[str]) -> list[int] | None:
    compact = raw.get("mask")
    if isinstance(compact, str):
        text = compact.strip()
        if len(text) != len(agents) or any(char not in {"0", "1"} for char in text):
            return None
        return [1 if char == "1" else 0 for char in text]

    legacy = raw.get("agent_mask")
    if not isinstance(legacy, list) or len(legacy) != len(agents):
        return None
    try:
        return [1 if int(value) else 0 for value in legacy]
    except (TypeError, ValueError):
        return None


def _reason_tag(mask: list[int]) -> str:
    total = sum(mask)
    if total == 0:
        return "rejected"
    if total == len(mask):
        return "broadcast"
    return "routed"


def _all_zero(memories: list[dict[str, Any]], agents: list[str], error: str) -> dict[str, Any]:
    return {
        "mask_matrix": [[0 for _ in agents] for _ in memories],
        "agent_score_matrix": [[0.0 for _ in agents] for _ in memories],
        "mask_reasons": [
            {
                "memory_id": str(memory.get("memory_id")),
                "reason_tag": "allocator_error",
            }
            for memory in memories
        ],
        "errors": [error],
    }


def reason_tag_counts(mask_reasons: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(item.get("reason_tag") or "unknown") for item in mask_reasons))
