"""Deterministic retrieval for reusable experience memories."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from mas_scope.core.types import TaskExample


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def load_memories(memory_files: list[str | Path]) -> list[dict[str, Any]]:
    memories: list[dict[str, Any]] = []
    for file_path in memory_files:
        path = Path(file_path)
        rows = _read_jsonl(path)
        namespace = _infer_namespace(path, rows)
        for line_no, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            original_id = str(row.get("memory_id") or f"{path.stem}_{line_no:06d}")
            memory = dict(row)
            memory["original_memory_id"] = original_id
            memory["memory_namespace"] = namespace
            memory["memory_id"] = f"{namespace}:{original_id}"
            memory["text"] = _memory_text(memory)
            memories.append(memory)
    return memories


def retrieve_top_k(example: TaskExample, memories: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    if top_k <= 0 or not memories:
        return []

    target_tokens = set(_tokenize(_target_text(example)))
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for memory in memories:
        score = _score_memory(example, memory, target_tokens)
        scored.append((score, str(memory.get("memory_id")), memory))
    scored.sort(key=lambda item: (-item[0], item[1]))

    retrieved: list[dict[str, Any]] = []
    for rank, (score, _, memory) in enumerate(scored[:top_k], start=1):
        item = dict(memory)
        item["retrieval_score"] = float(score)
        item["retrieval_rank"] = rank
        retrieved.append(item)
    return retrieved


def _read_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    if not path.exists():
        raise FileNotFoundError(f"Memory file does not exist: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _infer_namespace(path: Path, rows: list[dict[str, Any]]) -> str:
    name = path.stem.lower()
    if "strategyqa" in name:
        return "strategyqa"
    if "hotpotqa" in name or "hotpot" in name:
        return "hotpotqa"

    sample_text = " ".join(
        str(row.get("source_task_description", "")) + " " + str(row.get("condition", ""))
        for row in rows[:20]
        if isinstance(row, dict)
    ).lower()
    if "strategyqa" in sample_text:
        return "strategyqa"
    if "hotpotqa" in sample_text or "hotpot" in sample_text:
        return "hotpotqa"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_") or "memory"


def _memory_text(memory: dict[str, Any]) -> str:
    sections = []
    if memory.get("condition"):
        sections.append(f"Condition: {memory['condition']}")
    if memory.get("experience"):
        sections.append(f"Experience: {memory['experience']}")
    if memory.get("evidence"):
        sections.append(f"Evidence: {memory['evidence']}")
    if not sections:
        sections.append(str(memory.get("text") or memory.get("o_src") or ""))
    return "\n".join(part.strip() for part in sections if str(part).strip())


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


def _score_memory(example: TaskExample, memory: dict[str, Any], target_tokens: set[str]) -> float:
    memory_tokens = _tokenize(
        " ".join(
            str(memory.get(key, ""))
            for key in ("source_task_description", "condition", "experience", "evidence", "text", "memory_namespace")
        )
    )
    if not memory_tokens:
        return 0.0
    counts = Counter(memory_tokens)
    overlap = sum(min(counts[token], 2) for token in target_tokens if token in counts)
    length_norm = math.sqrt(len(memory_tokens))
    score = overlap / length_norm
    namespace = str(memory.get("memory_namespace") or "").lower()
    if namespace and namespace == example.dataset_name.lower():
        score += 2.0
    if example.task_type.lower() in counts:
        score += 0.5
    return score


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text) if len(token) > 1]
