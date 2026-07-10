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
BM25_K1 = 1.4
BM25_B = 0.75

FIELD_WEIGHTS = {
    "condition": 3,
    "experience": 3,
    "evidence": 2,
    "source_query": 1,
    "source_task_description": 1,
    "role": 1,
    "outcome": 1,
    "namespace": 1,
}

HOTPOTQA_SIGNALS = {
    "bridge",
    "comparison",
    "evidence",
    "alignment",
    "answer",
    "granularity",
    "yes",
    "no",
    "entity",
    "type",
    "critic",
    "actor",
    "summarizer",
}


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

    target_tokens = _tokenize(_target_text(example))
    target_counter = Counter(target_tokens)
    eligible_memories = [memory for memory in memories if not _same_source_query(example, memory)]
    memory_tokens_by_id = {
        str(memory.get("memory_id")): _memory_weighted_tokens(memory)
        for memory in eligible_memories
    }
    idf = _inverse_document_frequency(memory_tokens_by_id.values())
    avg_doc_len = _average_doc_len(memory_tokens_by_id.values())

    scored: list[tuple[float, str, dict[str, Any], list[str]]] = []
    for memory in eligible_memories:
        score, reasons = _score_memory(example, memory, target_counter, memory_tokens_by_id[str(memory.get("memory_id"))], idf, avg_doc_len)
        scored.append((score, str(memory.get("memory_id")), memory, reasons))
    scored.sort(key=lambda item: (-item[0], item[1]))

    retrieved: list[dict[str, Any]] = []
    for rank, (score, _, memory, reasons) in enumerate(scored[:top_k], start=1):
        item = dict(memory)
        item["retrieval_score"] = float(score)
        item["retrieval_rank"] = rank
        item["retrieval_reason"] = reasons
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
        str(_memory_dataset_name(row) or "")
        + " "
        + str(row.get("source_task_description", ""))
        + " "
        + str(row.get("condition", ""))
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


def _score_memory(
    example: TaskExample,
    memory: dict[str, Any],
    target_counter: Counter[str],
    memory_tokens: list[str],
    idf: dict[str, float],
    avg_doc_len: float,
) -> tuple[float, list[str]]:
    if not memory_tokens:
        return 0.0, ["empty_memory_text"]

    reasons: list[str] = []
    score = _bm25_score(target_counter, memory_tokens, idf, avg_doc_len)
    if score > 0:
        reasons.append("bm25_text_match")

    dataset_name = str(_memory_dataset_name(memory) or memory.get("memory_namespace") or "").lower()
    if dataset_name and dataset_name == example.dataset_name.lower():
        score += 3.0
        reasons.append("same_dataset")

    if example.task_type.lower() in set(memory_tokens):
        score += 0.5
        reasons.append("same_task_type")

    question_type = str(example.metadata.get("question_type") or "").lower()
    memory_token_set = set(memory_tokens)
    if question_type and question_type in memory_token_set:
        score += 1.0
        reasons.append(f"question_type:{question_type}")

    hotpot_overlap = sorted(HOTPOTQA_SIGNALS & set(target_counter) & memory_token_set)
    if example.dataset_name.lower() == "hotpotqa" and hotpot_overlap:
        boost = min(1.5, 0.25 * len(hotpot_overlap))
        score += boost
        reasons.append("hotpotqa_signals:" + ",".join(hotpot_overlap[:5]))

    role = _memory_role(memory)
    if role and role != "unknown":
        score += 0.2
        reasons.append(f"attributed_role:{role}")

    induction_type = _memory_induction_type(memory)
    if induction_type in {"success", "failure"}:
        score += 0.1
        reasons.append(f"induction:{induction_type}")

    return score, reasons


def _bm25_score(target_counter: Counter[str], memory_tokens: list[str], idf: dict[str, float], avg_doc_len: float) -> float:
    counts = Counter(memory_tokens)
    doc_len = len(memory_tokens)
    score = 0.0
    for token, query_tf in target_counter.items():
        tf = counts.get(token, 0)
        if tf <= 0:
            continue
        length_norm = 1 - BM25_B + BM25_B * (doc_len / avg_doc_len)
        bm25 = idf.get(token, 0.0) * (tf * (BM25_K1 + 1)) / (tf + BM25_K1 * length_norm)
        score += bm25 * min(float(query_tf), 3.0)
    return score


def _inverse_document_frequency(documents: Any) -> dict[str, float]:
    docs = [list(document) for document in documents]
    total_docs = len(docs)
    if total_docs == 0:
        return {}
    document_frequency: Counter[str] = Counter()
    for tokens in docs:
        document_frequency.update(set(tokens))
    return {
        token: max(0.0, math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5)))
        for token, freq in document_frequency.items()
    }


def _average_doc_len(documents: Any) -> float:
    lengths = [len(list(document)) for document in documents]
    if not lengths:
        return 1.0
    return max(1.0, sum(lengths) / len(lengths))


def _memory_weighted_tokens(memory: dict[str, Any]) -> list[str]:
    sections = {
        "condition": memory.get("condition"),
        "experience": memory.get("experience"),
        "evidence": memory.get("evidence"),
        "source_query": _memory_source(memory).get("query"),
        "source_task_description": memory.get("source_task_description"),
        "role": _memory_role(memory),
        "outcome": _memory_induction_type(memory),
        "namespace": memory.get("memory_namespace"),
    }
    tokens: list[str] = []
    for key, value in sections.items():
        section_tokens = _tokenize(str(value or ""))
        weight = FIELD_WEIGHTS.get(key, 1)
        for _ in range(weight):
            tokens.extend(section_tokens)
    if not tokens:
        tokens.extend(_tokenize(str(memory.get("text") or memory.get("o_src") or "")))
    return tokens


def _same_source_query(example: TaskExample, memory: dict[str, Any]) -> bool:
    source_query = _normalize_query(_memory_source(memory).get("query"))
    target_query = _normalize_query(_example_query(example))
    return bool(source_query and target_query and source_query == target_query)


def _example_query(example: TaskExample) -> str:
    if isinstance(example.input, dict):
        for key in ("question", "query", "task_description", "objective"):
            value = example.input.get(key)
            if value:
                return str(value)
    return json.dumps(example.input, ensure_ascii=False, sort_keys=True)


def _normalize_query(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def _memory_source(memory: dict[str, Any]) -> dict[str, Any]:
    source = memory.get("source")
    return source if isinstance(source, dict) else {}


def _memory_reasoningbank(memory: dict[str, Any]) -> dict[str, Any]:
    reasoningbank = memory.get("reasoningbank")
    return reasoningbank if isinstance(reasoningbank, dict) else {}


def _memory_dataset_name(memory: dict[str, Any]) -> Any:
    return memory.get("dataset_name") or _memory_source(memory).get("dataset_name")


def _memory_role(memory: dict[str, Any]) -> str:
    role = _memory_reasoningbank(memory).get("attributed_agent_role") or memory.get("r_src") or ""
    return str(role).strip().lower()


def _memory_induction_type(memory: dict[str, Any]) -> str:
    induction_type = _memory_reasoningbank(memory).get("induction_type")
    if induction_type:
        return str(induction_type).strip().lower()
    y_src = memory.get("y_src") or {}
    if isinstance(y_src, dict) and y_src.get("success") is True:
        return "success"
    if isinstance(y_src, dict) and y_src.get("success") is False:
        return "failure"
    return ""


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text) if len(token) > 1]
