"""Extract reusable experience memories from multi-agent trajectories."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_SRC = Path(__file__).resolve().parent / "src"
if REPO_SRC.exists():
    sys.path.insert(0, str(REPO_SRC))

from mas_scope.llm.openai_compatible import OpenAICompatibleLLM  # noqa: E402


FORBIDDEN_FIELDS = {
    "applicable_roles",
    "memory_type",
    "scope",
    "initial_scope",
    "mask",
    "agent_mask",
    "allocation_score",
    "target_agent",
    "target_role",
}

REQUIRED_OUTPUT_FIELDS = {
    "source_task_description",
    "condition",
    "experience",
    "evidence",
}


@dataclass(frozen=True)
class TrajectoryInput:
    trajectory_id: str
    source_file: str
    content: str
    source_line: int | None = None


def load_trajectories(input_dir: Path) -> list[TrajectoryInput]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    mas_trajectories = input_dir / "trajectories.jsonl"
    if mas_trajectories.exists():
        return _load_mas_run_dir(input_dir)

    trajectories: list[TrajectoryInput] = []
    for path in sorted(p for p in input_dir.rglob("*") if p.is_file()):
        suffix = path.suffix.lower()
        if suffix == ".json":
            trajectories.extend(_load_json_file(path))
        elif suffix == ".jsonl":
            trajectories.extend(_load_jsonl_file(path))
        else:
            text = _read_text(path)
            if text.strip():
                trajectories.append(
                    TrajectoryInput(
                        trajectory_id=path.stem,
                        source_file=str(path),
                        content=text,
                    )
                )
    return trajectories


def _load_mas_run_dir(input_dir: Path) -> list[TrajectoryInput]:
    examples_by_id = _load_jsonl_index(input_dir / "examples.jsonl", ("example_id", "id"))
    results_by_run_id = _load_jsonl_index(input_dir / "results.jsonl", ("run_id",))
    results_by_example_id = _load_jsonl_index(input_dir / "results.jsonl", ("example_id",))
    trajectories: list[TrajectoryInput] = []

    for trajectory in _load_jsonl_file(input_dir / "trajectories.jsonl"):
        try:
            payload = json.loads(trajectory.content)
        except json.JSONDecodeError:
            trajectories.append(trajectory)
            continue

        example_id = str(payload.get("example_id") or "")
        run_id = str(payload.get("run_id") or trajectory.trajectory_id)
        enriched = _compact_mas_run_payload(
            trajectory=payload,
            source_example=examples_by_id.get(example_id),
            result=results_by_run_id.get(run_id) or results_by_example_id.get(example_id),
        )
        trajectories.append(
            TrajectoryInput(
                trajectory_id=trajectory.trajectory_id,
                source_file=trajectory.source_file,
                source_line=trajectory.source_line,
                content=_serialize_payload(enriched),
            )
        )
    return trajectories


def _load_jsonl_index(path: Path, keys: tuple[str, ...]) -> dict[str, Any]:
    if not path.exists():
        return {}
    index: dict[str, Any] = {}
    for line in _read_text(path).splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        for key in keys:
            value = payload.get(key)
            if value is not None and str(value).strip():
                index[str(value)] = payload
                break
    return index


def _compact_mas_run_payload(
    trajectory: dict[str, Any],
    source_example: Any,
    result: Any,
) -> dict[str, Any]:
    compact_messages = []
    for message in trajectory.get("messages", []):
        if not isinstance(message, dict):
            continue
        compact_messages.append(
            {
                "turn_id": message.get("turn_id"),
                "agent_name": message.get("agent_name"),
                "role": message.get("role"),
                "content": message.get("content"),
            }
        )

    compact: dict[str, Any] = {
        "trajectory_id": trajectory.get("run_id") or trajectory.get("trajectory_id"),
        "example_id": trajectory.get("example_id"),
        "dataset_name": trajectory.get("dataset_name"),
        "mas_type": trajectory.get("mas_type"),
        "model_name": trajectory.get("model_name"),
        "task": _compact_source_example(source_example),
        "messages": compact_messages,
        "final_answer": trajectory.get("final_answer"),
        "error": trajectory.get("error"),
        "team_metadata": {
            "mas_style": (trajectory.get("metadata") or {}).get("mas_style"),
            "topology": (trajectory.get("metadata") or {}).get("topology"),
            "agent_order": (trajectory.get("metadata") or {}).get("agent_order"),
        },
        "result": _compact_result(result),
    }
    return compact


def _compact_source_example(source_example: Any) -> Any:
    if not isinstance(source_example, dict):
        return source_example
    task_input = source_example.get("input")
    if isinstance(task_input, dict):
        compact_input = {
            key: task_input.get(key)
            for key in ("question", "task_description", "objective", "facts", "decomposition")
            if key in task_input
        }
    else:
        compact_input = task_input
    if compact_input is None:
        compact_input = {
            key: source_example.get(key)
            for key in ("question", "task_description", "objective", "facts", "decomposition")
            if key in source_example
        } or None
    return {
        "example_id": source_example.get("example_id"),
        "dataset_name": source_example.get("dataset_name"),
        "task_type": source_example.get("task_type"),
        "input": compact_input,
        "target": source_example.get("target"),
    }


def _compact_result(result: Any) -> Any:
    if not isinstance(result, dict):
        return result
    return {
        "prediction": result.get("prediction"),
        "target": result.get("target"),
        "metrics": result.get("metrics"),
        "success": result.get("success"),
        "error": result.get("error"),
    }


def _load_json_file(path: Path) -> list[TrajectoryInput]:
    text = _read_text(path)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [TrajectoryInput(trajectory_id=path.stem, source_file=str(path), content=text)]

    if isinstance(payload, list):
        trajectories = []
        for index, item in enumerate(payload, start=1):
            fallback = f"{path.stem}_{index:04d}"
            trajectories.append(
                TrajectoryInput(
                    trajectory_id=_extract_trajectory_id(item, fallback),
                    source_file=str(path),
                    content=_serialize_payload(item),
                )
            )
        return trajectories

    return [
        TrajectoryInput(
            trajectory_id=_extract_trajectory_id(payload, path.stem),
            source_file=str(path),
            content=_serialize_payload(payload),
        )
    ]


def _load_jsonl_file(path: Path) -> list[TrajectoryInput]:
    trajectories: list[TrajectoryInput] = []
    for line_no, line in enumerate(_read_text(path).splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            content = _serialize_payload(payload)
            trajectory_id = _extract_trajectory_id(payload, f"{path.stem}_{line_no:06d}")
        except json.JSONDecodeError:
            content = line
            trajectory_id = f"{path.stem}_{line_no:06d}"
        trajectories.append(
            TrajectoryInput(
                trajectory_id=trajectory_id,
                source_file=str(path),
                content=content,
                source_line=line_no,
            )
        )
    return trajectories


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _serialize_payload(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _extract_trajectory_id(payload: Any, fallback: str) -> str:
    if isinstance(payload, dict):
        for key in ("trajectory_id", "run_id", "example_id", "id"):
            value = payload.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return fallback


def build_messages(trajectory: TrajectoryInput, max_chars: int) -> list[dict[str, str]]:
    content = trajectory.content
    if max_chars > 0 and len(content) > max_chars:
        content = content[:max_chars] + "\n...[TRUNCATED]"

    system_prompt = """You extract reusable experience memories from multi-agent task trajectories.

Return only a valid JSON array. Do not wrap it in Markdown.
Each array item must follow this exact schema:
{
  "source": {
    "producer_agent": "agent name or multi-agent or unknown",
    "producer_role": "agent role or team or unknown"
  },
  "source_task_description": "The source task, subtask, or local situation where this experience was observed.",
  "condition": "When this experience may be applicable in future tasks.",
  "experience": "A reusable lesson, strategy, warning, or heuristic distilled from the trajectory.",
  "evidence": "The key source behavior, outcome, failure, correction, or feedback that supports this experience."
}

Rules:
- Extract 0 to 3 memories for this trajectory.
- Extract only reusable experience that can help future agent decisions.
- Prefer multi-agent execution lessons: decomposition, role responsibility, coordination, handoff, critique, tool-use, aggregation, communication, failure correction.
- Do not output ordinary facts, final answers, complete trajectory summaries, or task-specific answers.
- source.producer_agent and source.producer_role are provenance only. They describe where the experience came from, not who should receive it in the future.
- If the experience is about team coordination rather than one agent, use producer_agent="multi-agent" and producer_role="team".
- If producer_agent is a concrete agent, producer_role must be that agent's source role, not "team".
- If the source cannot be determined, use producer_agent="unknown" and producer_role="unknown".
- If a trajectory succeeds but exposes a process weakness, state both the success and the process weakness in evidence.
- Do not include future allocation fields such as applicable_roles, memory_type, scope, initial_scope, mask, agent_mask, allocation_score, target_agent, or target_role.
"""
    user_prompt = f"""Trajectory id: {trajectory.trajectory_id}
Source file: {trajectory.source_file}

Trajectory:
{content}
"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_json_array(raw_response: str) -> list[Any]:
    text = raw_response.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        fenced_text = fenced.group(1).strip()
        parsed = json.loads(fenced_text)
        if isinstance(parsed, list):
            return parsed
        raise ValueError("Fenced JSON content is not an array.")

    array_text = _extract_first_json_array(text)
    parsed = json.loads(array_text)
    if not isinstance(parsed, list):
        raise ValueError("JSON content is not an array.")
    return parsed


def _extract_first_json_array(text: str) -> str:
    start = text.find("[")
    if start < 0:
        raise ValueError("No JSON array found in response.")

    in_string = False
    escaped = False
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError("Unclosed JSON array in response.")


def sanitize_memory(item: Any, trajectory: TrajectoryInput, memory_index: int) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    source = item.get("source")
    if not isinstance(source, dict):
        source = item

    producer_agent = _clean_text(source.get("producer_agent")) or "unknown"
    producer_role = _clean_text(source.get("producer_role")) or "unknown"
    if producer_agent not in {"multi-agent", "unknown"} and producer_role == "team":
        producer_role = producer_agent

    sanitized = {
        "memory_id": f"m_{memory_index:06d}",
        "source": {
            "trajectory_id": trajectory.trajectory_id,
            "producer_agent": producer_agent,
            "producer_role": producer_role,
        },
        "source_task_description": _clean_text(item.get("source_task_description")),
        "condition": _clean_text(item.get("condition")),
        "experience": _clean_text(item.get("experience")),
        "evidence": _clean_text(item.get("evidence")),
    }

    if any(not sanitized[field] for field in REQUIRED_OUTPUT_FIELDS):
        return None
    return sanitized


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False).strip()


def write_failed_response(
    logs_dir: Path,
    trajectory: TrajectoryInput,
    raw_response: str,
    error: str,
) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", trajectory.trajectory_id)[:120] or "unknown"
    path = logs_dir / f"{safe_id}.json"
    suffix = 1
    while path.exists():
        path = logs_dir / f"{safe_id}_{suffix}.json"
        suffix += 1
    payload = {
        "trajectory_id": trajectory.trajectory_id,
        "source_file": trajectory.source_file,
        "source_line": trajectory.source_line,
        "error": error,
        "raw_response": raw_response,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_extraction(
    input_dir: Path,
    output_file: Path,
    llm: Any,
    logs_dir: Path,
    limit: int | None = None,
    overwrite: bool = False,
    max_chars: int = 60000,
    max_tokens: int = 2000,
) -> dict[str, int]:
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists. Pass --overwrite to replace it: {output_file}")

    trajectories = load_trajectories(input_dir)
    if limit is not None:
        trajectories = trajectories[:limit]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    memory_index = 1
    stats = {
        "trajectories": len(trajectories),
        "memories": 0,
        "failed": 0,
        "dropped": 0,
    }

    with output_file.open("w", encoding="utf-8", newline="\n") as handle:
        for trajectory in trajectories:
            messages = build_messages(trajectory, max_chars=max_chars)
            raw_response = ""
            try:
                response = llm.generate(messages, temperature=0.0, max_tokens=max_tokens)
                raw_response = response.content
                parsed_items = parse_json_array(raw_response)
            except Exception as exc:
                stats["failed"] += 1
                write_failed_response(logs_dir, trajectory, raw_response, str(exc))
                continue

            for item in parsed_items[:3]:
                memory = sanitize_memory(item, trajectory, memory_index)
                if memory is None:
                    stats["dropped"] += 1
                    continue
                handle.write(json.dumps(memory, ensure_ascii=False) + "\n")
                memory_index += 1
                stats["memories"] += 1

    return stats


def create_llm(args: argparse.Namespace) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        model_name=args.model,
        timeout=args.timeout,
        retries=args.retries,
        temperature=0.0,
        max_tokens=args.max_tokens,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract reusable experience memories from trajectories.")
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_file", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--logs_dir", default=Path("logs/failed_responses"), type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_chars", type=int, default=60000)
    parser.add_argument("--max_tokens", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm = create_llm(args)
    stats = run_extraction(
        input_dir=args.input_dir,
        output_file=args.output_file,
        llm=llm,
        logs_dir=args.logs_dir,
        limit=args.limit,
        overwrite=args.overwrite,
        max_chars=args.max_chars,
        max_tokens=args.max_tokens,
    )
    print(
        "Extraction complete: "
        f"trajectories={stats['trajectories']} "
        f"memories={stats['memories']} "
        f"failed={stats['failed']} "
        f"dropped={stats['dropped']}"
    )


if __name__ == "__main__":
    main()
