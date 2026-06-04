"""Diagnostics for allocator ablation runs."""

from __future__ import annotations

from collections import Counter
from typing import Any


def compute_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"num_records": 0}

    retrieval_counts = [len(record.get("retrieved_memories") or []) for record in records]
    mask_stats = _masking_stats(records)
    selection_stats = _selection_stats(records)
    realization_stats = _realization_stats(records)
    task_stats = _task_stats(records)
    allocator_error_records = sum(1 for record in records if record.get("allocator_errors"))

    return {
        "num_records": len(records),
        "retrieval": {
            "avg_retrieved_memory_count": _avg(retrieval_counts),
            "top_k_values": sorted({record.get("retrieval_top_k") for record in records if record.get("retrieval_top_k") is not None}),
            "masking_reject_ratio": mask_stats["masking_reject_ratio"],
        },
        "masking": mask_stats,
        "selection": selection_stats,
        "realization": realization_stats,
        "allowed_cap": _allowed_cap_stats(records),
        "task": task_stats,
        "allocator_error_rate": allocator_error_records / len(records),
        "allocator_error_count": allocator_error_records,
    }


def paired_comparison(records_by_setting: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, int]]:
    comparisons: dict[str, dict[str, int]] = {}
    for left, right in (("B0", "B1"), ("B1", "B2"), ("B2", "B3"), ("B3", "B4"), ("B2", "G2"), ("B3", "G3"), ("G2", "G3")):
        left_scores = _scores_by_target(records_by_setting.get(left, []))
        right_scores = _scores_by_target(records_by_setting.get(right, []))
        shared = sorted(set(left_scores) & set(right_scores))
        stats = {"shared": len(shared), "improved": 0, "worse": 0, "unchanged": 0}
        for target_id in shared:
            delta = right_scores[target_id] - left_scores[target_id]
            if delta > 1e-9:
                stats["improved"] += 1
            elif delta < -1e-9:
                stats["worse"] += 1
            else:
                stats["unchanged"] += 1
        comparisons[f"{right}_vs_{left}"] = stats
    return comparisons


def _masking_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    total_cells = 0
    one_cells = 0
    row_count = 0
    all_zero_rows = 0
    all_one_rows = 0
    agents_per_memory: list[int] = []
    allowed_per_agent: list[int] = []
    reason_tags: Counter[str] = Counter()

    for record in records:
        agents = record.get("agents") or []
        matrix = record.get("mask_matrix") or []
        if agents:
            col_counts = [0 for _ in agents]
        else:
            col_counts = []
        for row in matrix:
            clean = [1 if int(value) else 0 for value in row]
            row_count += 1
            total_cells += len(clean)
            row_sum = sum(clean)
            one_cells += row_sum
            agents_per_memory.append(row_sum)
            if row_sum == 0:
                all_zero_rows += 1
            if clean and row_sum == len(clean):
                all_one_rows += 1
            for col, value in enumerate(clean):
                if col < len(col_counts):
                    col_counts[col] += value
        allowed_per_agent.extend(col_counts)
        for reason in record.get("mask_reasons") or []:
            reason_tags[str(reason.get("reason_tag") or "unknown")] += 1

    density = one_cells / total_cells if total_cells else 0.0
    return {
        "mask_density": density,
        "masking_reject_ratio": 1.0 - density if total_cells else 0.0,
        "all_zero_memory_rate": all_zero_rows / row_count if row_count else 0.0,
        "all_one_memory_rate": all_one_rows / row_count if row_count else 0.0,
        "avg_agents_per_memory": _avg(agents_per_memory),
        "avg_allowed_memories_per_agent": _avg(allowed_per_agent),
        "reason_tag_counts": dict(reason_tags),
    }


def _selection_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    selected_counts: list[int] = []
    ratios: list[float] = []
    redundancy_rates: list[float] = []
    block_lengths: list[int] = []

    for record in records:
        agents = record.get("agents") or []
        mask = record.get("mask_matrix") or []
        selected = record.get("selected_matrix") or []
        realized = record.get("realized_memories") or {}
        for col, agent in enumerate(agents):
            allowed = sum(1 for row in mask if col < len(row) and int(row[col]) == 1)
            chosen = sum(1 for row in selected if col < len(row) and int(row[col]) == 1)
            selected_counts.append(chosen)
            ratios.append(chosen / allowed if allowed else 0.0)
            texts = [str(item.get("text") or "") for item in realized.get(agent, [])]
            block_lengths.append(sum(len(text) for text in texts))
            redundancy_rates.append(_redundancy_rate(texts))

    return {
        "avg_final_memory_count_per_agent": _avg(selected_counts),
        "selected_allowed_ratio": _avg(ratios),
        "simple_redundancy_rate": _avg(redundancy_rates),
        "avg_memory_block_length": _avg(block_lengths),
    }


def _realization_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    type_counts: Counter[str] = Counter()
    lengths: list[int] = []
    for record in records:
        realized = record.get("realized_memories") or {}
        for items in realized.values():
            for item in items:
                type_counts[str(item.get("realization_type") or "raw")] += 1
                lengths.append(len(str(item.get("text") or "")))
    return {
        "realization_type_counts": dict(type_counts),
        "avg_realized_memory_length": _avg(lengths),
    }


def _allowed_cap_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    enabled = [record.get("allowed_cap_stats") or {} for record in records if (record.get("allowed_cap_stats") or {}).get("enabled")]
    if not enabled:
        return {
            "enabled_record_count": 0,
            "allowed_cap_dropped_count": 0,
            "allowed_cap_dropped_ratio": 0.0,
            "avg_allowed_before_cap": 0.0,
            "avg_allowed_after_cap": 0.0,
        }
    return {
        "enabled_record_count": len(enabled),
        "allowed_cap_dropped_count": sum(int(item.get("dropped_count") or 0) for item in enabled),
        "allowed_cap_dropped_ratio": _avg([float(item.get("dropped_ratio") or 0.0) for item in enabled]),
        "avg_allowed_before_cap": _avg([float(item.get("avg_allowed_before_cap") or 0.0) for item in enabled]),
        "avg_allowed_after_cap": _avg([float(item.get("avg_allowed_after_cap") or 0.0) for item in enabled]),
    }


def _task_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [_task_score(record.get("task_result") or {}) for record in records]
    successes = [1 if (record.get("task_result") or {}).get("success") else 0 for record in records]
    failure_types: Counter[str] = Counter()
    for record in records:
        result = record.get("task_result") or {}
        if result.get("success"):
            failure_types["success"] += 1
        elif result.get("error"):
            failure_types["error"] += 1
        else:
            failure_types["metric_failure"] += 1
    return {
        "success_rate": _avg(successes),
        "average_score": _avg(scores),
        "failure_type_distribution": dict(failure_types),
    }


def _scores_by_target(records: list[dict[str, Any]]) -> dict[str, float]:
    return {
        str(record.get("target_task_id")): _task_score(record.get("task_result") or {})
        for record in records
    }


def _task_score(result: dict[str, Any]) -> float:
    metrics = result.get("metrics") or {}
    for key in ("exact_match", "yes_no_accuracy", "success", "valid_action_format"):
        if key in metrics and isinstance(metrics[key], (int, float, bool)):
            return float(metrics[key])
    numeric = [float(value) for value in metrics.values() if isinstance(value, (int, float, bool))]
    if numeric:
        return sum(numeric) / len(numeric)
    return 1.0 if result.get("success") else 0.0


def _redundancy_rate(texts: list[str]) -> float:
    if len(texts) < 2:
        return 0.0
    overlaps = []
    token_sets = [set(_tokens(text)) for text in texts]
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            union = token_sets[i] | token_sets[j]
            if not union:
                overlaps.append(0.0)
            else:
                overlaps.append(len(token_sets[i] & token_sets[j]) / len(union))
    return _avg(overlaps)


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in text.split() if len(token) > 2]


def _avg(values: list[float] | list[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
