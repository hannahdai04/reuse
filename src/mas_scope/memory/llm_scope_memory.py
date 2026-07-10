"""Agent-mask memory routing provider."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from mas_scope.core.registry import registry
from mas_scope.core.types import AgentSpec, TaskExample
from mas_scope.llm.mock import MockLLM
from mas_scope.llm.openai_compatible import OpenAICompatibleLLM
from mas_scope.memory.base import MemoryProvider


VALID_POLICIES = {"reuse", "abstract-then-reuse", "deny"}
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@registry.register_memory_provider("llm-scope")
class LLMScopeMemoryProvider(MemoryProvider):
    """Routes reusable memories to concrete agents with a binary agent mask."""

    def __init__(
        self,
        bank_path: str | Path | None = None,
        policy_mode: str = "all-agents",
        candidate_top_k: int = 20,
        selected_top_k: int = 3,
        score_threshold: float = 0.6,
        policy_llm: dict | None = None,
        **kwargs,
    ) -> None:
        self.bank_path = Path(bank_path) if bank_path else None
        self.policy_mode = policy_mode
        self.candidate_top_k = candidate_top_k
        self.selected_top_k = selected_top_k
        self.score_threshold = score_threshold
        self.policy_llm_config = policy_llm or {}
        self.memories = self._load_bank(self.bank_path)
        self.last_retrieval_metadata: dict = {}
        self._decision_cache: dict[tuple[str, tuple[str, ...]], dict] = {}
        self._policy_llm = None

    def retrieve(self, example: TaskExample, agent_spec: AgentSpec, context: dict) -> list[dict]:
        agent_order = list(context.get("agent_order") or [agent_spec.role])
        current_agent = str(context.get("current_agent") or agent_spec.role)
        errors: list[str] = []
        if current_agent not in agent_order:
            self.last_retrieval_metadata = self._metadata([], [], agent_order, current_agent, errors + ["current_agent_not_in_agent_order"])
            return []

        cache_key = (example.example_id, tuple(agent_order))
        if cache_key not in self._decision_cache:
            candidates = self._recall_candidates(example, agent_order)
            decisions, decision_errors = self._decide(example, candidates, agent_order, context)
            self._decision_cache[cache_key] = {
                "candidates": candidates,
                "decisions": decisions,
                "errors": decision_errors,
                "policy_prompt_chars": context.get("last_policy_prompt_chars"),
                "policy_candidate_count": context.get("last_policy_candidate_count"),
            }

        cached = self._decision_cache[cache_key]
        candidates = cached["candidates"]
        decisions = cached["decisions"]
        errors.extend(cached.get("errors", []))
        policy_prompt_chars = cached.get("policy_prompt_chars")
        policy_candidate_count = cached.get("policy_candidate_count")
        agent_index = agent_order.index(current_agent)
        memory_by_id = {str(memory.get("memory_id")): memory for memory in candidates}

        selected: list[dict] = []
        for decision in sorted(decisions, key=lambda item: float(item.get("score", 0.0)), reverse=True):
            mask = decision.get("agent_mask") or []
            if agent_index >= len(mask) or int(mask[agent_index]) != 1:
                continue
            if float(decision.get("score", 0.0)) < self.score_threshold:
                continue
            memory = memory_by_id.get(str(decision.get("memory_id")))
            if not memory:
                continue
            selected_memory = dict(memory)
            selected_memory["decision"] = decision
            selected.append(selected_memory)
            if len(selected) >= self.selected_top_k:
                break

        self.last_retrieval_metadata = self._metadata(
            candidates,
            decisions,
            agent_order,
            current_agent,
            errors,
            selected,
            policy_prompt_chars=policy_prompt_chars,
            policy_candidate_count=policy_candidate_count,
        )
        return selected

    def update(self, example: TaskExample, trajectory, result) -> None:
        return None

    def _load_bank(self, bank_path: Path | None) -> list[dict]:
        if bank_path is None or not bank_path.exists():
            return []
        memories: list[dict] = []
        with bank_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "memory_id" not in payload:
                    payload["memory_id"] = f"{bank_path.stem}_{line_no}"
                memories.append(payload)
        return memories

    def _recall_candidates(self, example: TaskExample, agent_order: list[str]) -> list[dict]:
        scored: list[tuple[float, dict]] = []
        target_tokens = set(_tokenize(_target_text(example)))
        for memory in self.memories:
            if str(memory.get("source_example_id")) == example.example_id:
                continue
            source_query = _memory_source(memory).get("query")
            if source_query and str(source_query).strip() == str(example.input.get("question") or "").strip():
                continue
            score = self._candidate_score(example, memory, agent_order, target_tokens)
            scored.append((score, memory))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [memory for _, memory in scored[: self.candidate_top_k]]

    def _candidate_score(self, example: TaskExample, memory: dict, agent_order: list[str], target_tokens: set[str] | None = None) -> float:
        score = 0.0
        if _memory_task_type(memory) == example.task_type:
            score += 4.0
        if _memory_dataset_name(memory) == example.dataset_name:
            score += 3.0
        source_role = _memory_source_role(memory)
        if _role_matches_any_agent(source_role, agent_order):
            score += 2.0
        if _memory_success(memory) is True:
            score += 1.0
        if memory.get("p", {}).get("environment") == example.metadata.get("environment"):
            score += 0.5
        text_tokens = _tokenize(_memory_text(memory))
        if target_tokens and text_tokens:
            overlap = len(set(text_tokens) & target_tokens)
            score += overlap / math.sqrt(len(text_tokens))
        return score

    def _decide(self, example: TaskExample, candidates: list[dict], agent_order: list[str], context: dict) -> tuple[list[dict], list[str]]:
        if self.policy_mode == "all-agents":
            return [self._decision(memory, 1.0, [1] * len(agent_order), "reuse", "all agents receive this memory") for memory in candidates], []
        if self.policy_mode == "source-role-match":
            return [self._source_role_decision(memory, agent_order) for memory in candidates], []
        if self.policy_mode == "assistant-only":
            mask = [1 if "assistant" in role.lower() else 0 for role in agent_order]
            return [self._decision(memory, 1.0, mask, "reuse" if any(mask) else "deny", "assistant-only routing") for memory in candidates], []
        if self.policy_mode == "llm-mask":
            return self._llm_mask_decisions(example, candidates, agent_order, context)
        return [], [f"unknown_policy_mode:{self.policy_mode}"]

    def _source_role_decision(self, memory: dict, agent_order: list[str]) -> dict:
        source_role = _memory_source_role(memory)
        if source_role == "team":
            mask = [1 for _ in agent_order]
        else:
            mask = [1 if _role_matches_agent(source_role, role) else 0 for role in agent_order]
        policy = "reuse" if any(mask) else "deny"
        return self._decision(memory, 1.0 if any(mask) else 0.0, mask, policy, "source role matches receiving agent")

    def _decision(self, memory: dict, score: float, agent_mask: list[int], policy: str, reason: str) -> dict:
        if not any(agent_mask):
            policy = "deny"
        return {
            "memory_id": str(memory.get("memory_id")),
            "score": float(score),
            "agent_mask": [1 if int(value) else 0 for value in agent_mask],
            "policy": policy if policy in VALID_POLICIES else "reuse",
            "reason": reason,
        }

    def _llm_mask_decisions(
        self,
        example: TaskExample,
        candidates: list[dict],
        agent_order: list[str],
        context: dict,
    ) -> tuple[list[dict], list[str]]:
        if not candidates:
            return [], []
        try:
            llm = self._get_policy_llm()
            user_prompt = self._policy_user_prompt(example, candidates, agent_order, context)
            context["last_policy_prompt_chars"] = len(user_prompt)
            context["last_policy_candidate_count"] = len(candidates)
            response = llm.generate(
                [
                    {"role": "system", "content": self._policy_system_prompt()},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.policy_llm_config.get("max_tokens", 256),
                temperature=self.policy_llm_config.get("temperature", 0),
                extra_body=self.policy_llm_config.get("extra_body", {}),
            )
            payload = self._parse_json_object(response.content)
            return self._validate_llm_payload(payload, candidates, agent_order)
        except Exception as exc:
            return [], [f"llm_mask_error:{exc}"]

    def _get_policy_llm(self):
        if self._policy_llm is not None:
            return self._policy_llm
        provider = self.policy_llm_config.get("provider", "openai-compatible")
        if provider == "mock":
            self._policy_llm = MockLLM(model_name=self.policy_llm_config.get("model", "mock-llm"))
            return self._policy_llm
        self._policy_llm = OpenAICompatibleLLM(
            model_name=self.policy_llm_config.get("model"),
            timeout=self.policy_llm_config.get("timeout", 30.0),
            retries=self.policy_llm_config.get("retries", 0),
            temperature=self.policy_llm_config.get("temperature", 0.0),
            max_tokens=self.policy_llm_config.get("max_tokens", 256),
            extra_body=self.policy_llm_config.get("extra_body", {}),
        )
        return self._policy_llm

    def _policy_system_prompt(self) -> str:
        return (
            "You are a memory routing policy for a multi-agent system. "
            "Return only valid JSON. Select which candidate memories should be used. "
            "Use a binary memory_mask aligned exactly with candidate_memories."
        )

    def _policy_user_prompt(self, example: TaskExample, candidates: list[dict], agent_order: list[str], context: dict) -> str:
        candidate_payload = [
            {
                "memory_id": memory.get("memory_id"),
                "source_example_id": memory.get("source_example_id"),
                "source_query": _memory_source(memory).get("query"),
                "dataset_name": _memory_dataset_name(memory),
                "task_type": _memory_task_type(memory),
                "source_role": _memory_source_role(memory),
                "induction_type": _memory_induction_type(memory),
                "success": _memory_success(memory),
                "text": _memory_text(memory)[:450],
            }
            for memory in candidates
        ]
        target_context = {
            "example_id": example.example_id,
            "dataset_name": example.dataset_name,
            "task_type": example.task_type,
            "input": example.input,
            "agent_order": agent_order,
            "current_history": context.get("graph") or context.get("history") or {},
        }
        return json.dumps(
            {
                "target_context": target_context,
                "candidate_memories": candidate_payload,
                "required_output": {
                    "memory_mask": [0 for _ in candidate_payload],
                },
            },
            ensure_ascii=False,
        )

    def _parse_json_object(self, content: str) -> dict:
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end < start:
                raise
            return json.loads(text[start : end + 1])

    def _validate_llm_payload(self, payload: dict, candidates: list[dict], agent_order: list[str]) -> tuple[list[dict], list[str]]:
        candidate_ids = {str(memory.get("memory_id")): memory for memory in candidates}
        decisions: list[dict] = []
        errors: list[str] = []
        memory_mask = payload.get("memory_mask")
        if isinstance(memory_mask, str):
            memory_mask = [char for char in memory_mask.strip() if char in {"0", "1"}]
        if isinstance(memory_mask, list):
            if len(memory_mask) != len(candidates):
                errors.append("memory_mask_length_mismatch")
                return [], errors
            for memory, raw_value in zip(candidates, memory_mask):
                try:
                    value = int(raw_value)
                except (TypeError, ValueError):
                    errors.append(f"invalid_memory_mask_value:{memory.get('memory_id')}")
                    continue
                if value:
                    decisions.append(self._decision(memory, 1.0, [1] * len(agent_order), "reuse", "selected by memory_mask"))
            return decisions, errors

        if payload.get("agent_order") != agent_order:
            errors.append("agent_order_mismatch")
        for raw in payload.get("decisions", []):
            memory_id = str(raw.get("memory_id"))
            memory = candidate_ids.get(memory_id)
            if memory is None:
                errors.append(f"unknown_memory_id:{memory_id}")
                continue
            mask = raw.get("agent_mask")
            if not isinstance(mask, list) or len(mask) != len(agent_order):
                errors.append(f"invalid_mask:{memory_id}")
                continue
            try:
                clean_mask = [1 if int(value) else 0 for value in mask]
                score = float(raw.get("score", 0.0))
            except (TypeError, ValueError):
                errors.append(f"invalid_score_or_mask:{memory_id}")
                continue
            policy = str(raw.get("policy") or "reuse")
            if policy not in VALID_POLICIES:
                policy = "reuse"
            if policy == "deny":
                clean_mask = [0 for _ in agent_order]
            decisions.append(
                self._decision(
                    memory,
                    score,
                    clean_mask,
                    policy,
                    str(raw.get("reason") or ""),
                )
            )
        return decisions, errors

    def _metadata(
        self,
        candidates: list[dict],
        decisions: list[dict],
        agent_order: list[str],
        current_agent: str,
        errors: list[str],
        selected: list[dict] | None = None,
        policy_prompt_chars: int | None = None,
        policy_candidate_count: int | None = None,
    ) -> dict:
        selected = selected or []
        return {
            "candidate_count": len(candidates),
            "policy_candidate_count": policy_candidate_count,
            "policy_prompt_chars": policy_prompt_chars,
            "selected_ids": [str(memory.get("memory_id")) for memory in selected],
            "agent_order": agent_order,
            "current_agent": current_agent,
            "agent_mask_rows": [decision.get("agent_mask", []) for decision in decisions],
            "decisions": decisions,
            "policy_mode": self.policy_mode,
            "errors": errors,
        }


def _memory_source(memory: dict) -> dict:
    source = memory.get("source")
    return source if isinstance(source, dict) else {}


def _memory_reasoningbank(memory: dict) -> dict:
    reasoningbank = memory.get("reasoningbank")
    return reasoningbank if isinstance(reasoningbank, dict) else {}


def _memory_dataset_name(memory: dict) -> Any:
    return memory.get("dataset_name") or _memory_source(memory).get("dataset_name")


def _memory_task_type(memory: dict) -> Any:
    return memory.get("task_type")


def _memory_source_role(memory: dict) -> str:
    role = _memory_reasoningbank(memory).get("attributed_agent_role") or memory.get("r_src")
    return str(role or "").strip().lower()


def _memory_induction_type(memory: dict) -> str | None:
    value = _memory_reasoningbank(memory).get("induction_type")
    return str(value).strip().lower() if value else None


def _memory_success(memory: dict) -> bool | None:
    y_src = memory.get("y_src") or {}
    if isinstance(y_src, dict) and isinstance(y_src.get("success"), bool):
        return y_src.get("success")
    induction_type = _memory_induction_type(memory)
    if induction_type == "success":
        return True
    if induction_type == "failure":
        return False
    return None


def _memory_text(memory: dict) -> str:
    parts = []
    for key in ("condition", "experience", "evidence"):
        value = str(memory.get(key) or "").strip()
        if value:
            parts.append(f"{key}: {value}")
    if not parts:
        value = str(memory.get("text") or memory.get("o_src") or "").strip()
        if value:
            parts.append(value)
    return "\n".join(parts)


def _target_text(example: TaskExample) -> str:
    return json.dumps(
        {
            "dataset_name": example.dataset_name,
            "task_type": example.task_type,
            "input": example.input,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text) if len(token) > 1]


def _role_matches_any_agent(source_role: str, agent_order: list[str]) -> bool:
    if source_role == "team":
        return True
    return any(_role_matches_agent(source_role, agent_role) for agent_role in agent_order)


def _role_matches_agent(source_role: str, agent_role: str) -> bool:
    source = source_role.lower().strip()
    target = agent_role.lower().strip()
    return bool(source and (source == target or source in target or target in source))
