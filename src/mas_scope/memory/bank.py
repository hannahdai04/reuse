"""Utilities for extracting reusable memory units from run artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from mas_scope.core.ids import safe_slug


def build_memory_bank(run_dir: str | Path, output: str | Path, append: bool = False) -> int:
    run_path = Path(run_dir)
    output_path = Path(output)
    examples = _read_jsonl_by_key(run_path / "examples.jsonl", "example_id")
    results = _read_jsonl_by_key(run_path / "results.jsonl", "example_id")
    trajectories = _read_jsonl(run_path / "trajectories.jsonl")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    count = 0
    with output_path.open(mode, encoding="utf-8") as handle:
        for trajectory in trajectories:
            example_id = str(trajectory.get("example_id"))
            example = examples.get(example_id, {})
            result = results.get(example_id, {})
            for message in trajectory.get("messages", []):
                memory = _memory_from_message(run_path, trajectory, example, result, message)
                handle.write(json.dumps(memory, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
    return count


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing run artifact: {path}")
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _read_jsonl_by_key(path: Path, key: str) -> dict[str, dict]:
    return {str(record.get(key)): record for record in _read_jsonl(path)}


def _memory_from_message(run_path: Path, trajectory: dict, example: dict, result: dict, message: dict) -> dict:
    example_id = str(trajectory.get("example_id"))
    role = str(message.get("role") or message.get("agent_name") or "agent")
    turn_id = message.get("turn_id")
    memory_id = "_".join(
        [
            safe_slug(run_path.name),
            safe_slug(example_id),
            safe_slug(role),
            str(turn_id),
        ]
    )
    source_input = example.get("input") or {}
    text = _memory_text(source_input, role, message.get("content"))
    success = bool(result.get("success"))
    return {
        "memory_id": memory_id,
        "source_run_id": trajectory.get("run_id"),
        "source_run_dir": str(run_path),
        "source_example_id": example_id,
        "dataset_name": trajectory.get("dataset_name") or example.get("dataset_name"),
        "task_type": example.get("task_type") or trajectory.get("metadata", {}).get("task_type"),
        "c_src": _source_context(source_input),
        "a_src": message.get("agent_name"),
        "r_src": role,
        "o_src": message.get("content"),
        "y_src": {
            "success": success,
            "metrics": result.get("metrics", {}),
            "error": result.get("error"),
        },
        "p": {
            "source_run_dir": str(run_path),
            "mas_type": trajectory.get("mas_type"),
            "model_name": trajectory.get("model_name"),
            "turn_id": turn_id,
            "environment": example.get("metadata", {}).get("environment"),
            "reliability": 1.0 if success else 0.25,
            "access": "shared",
        },
        "text": text,
    }


def _source_context(source_input: dict) -> str:
    for key in ("question", "instruction", "goal"):
        value = source_input.get(key)
        if value:
            return str(value)
    return json.dumps(source_input, ensure_ascii=False, sort_keys=True)[:1000]


def _memory_text(source_input: dict, role: str, content) -> str:
    source_context = _source_context(source_input)
    content_text = str(content or "").strip()
    if len(content_text) > 1200:
        content_text = content_text[:1200] + "...[truncated]"
    return f"Source task: {source_context}\nSource role: {role}\nObserved output: {content_text}"
