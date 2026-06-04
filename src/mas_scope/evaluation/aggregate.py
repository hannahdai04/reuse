"""Metric aggregation."""

from __future__ import annotations

from mas_scope.core.types import TaskResult


def aggregate_metrics(results: list[TaskResult]) -> dict:
    if not results:
        return {"num_examples": 0}
    aggregate = {
        "num_examples": len(results),
        "success_rate": sum(1 for result in results if result.success) / len(results),
    }
    buckets: dict[str, list[float]] = {}
    for result in results:
        for key, value in result.metrics.items():
            if key == "scored":
                continue
            if isinstance(value, bool):
                buckets.setdefault(key, []).append(float(value))
            elif isinstance(value, (int, float)):
                buckets.setdefault(key, []).append(float(value))
    for key, values in buckets.items():
        aggregate[f"mean_{key}"] = sum(values) / len(values)
    return aggregate
