"""Extract reusable experience memories from multi-agent trajectories."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
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
    metadata: dict[str, Any] = field(default_factory=dict)


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
                metadata=_trajectory_metadata(enriched),
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
            "edges": (trajectory.get("metadata") or {}).get("edges"),
            "agent_order": (trajectory.get("metadata") or {}).get("agent_order"),
        },
        "result": _compact_result(result),
    }
    return compact


def _compact_source_example(source_example: Any) -> Any:
    if not isinstance(source_example, dict):
        return source_example
    task_input = source_example.get("input")
    target = source_example.get("target")
    dataset_name = source_example.get("dataset_name")
    if isinstance(task_input, dict):
        compact_input = _compact_task_input(task_input, target, dataset_name)
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
        "dataset_name": dataset_name,
        "task_type": source_example.get("task_type"),
        "input": compact_input,
        "target": target,
        "metadata": _compact_example_metadata(source_example.get("metadata")),
    }


def _compact_task_input(source_input: dict[str, Any], target: Any, dataset_name: Any) -> dict[str, Any]:
    compact = {
        key: source_input.get(key)
        for key in ("question", "task_description", "objective", "facts", "decomposition", "evidence")
        if key in source_input
    }
    context_summary = _compact_context_summary(source_input.get("context"), target)
    if context_summary:
        compact["context_summary"] = context_summary
    if "supporting_facts" in source_input and "supporting_facts" not in compact:
        compact["supporting_facts"] = source_input.get("supporting_facts")
    if str(dataset_name or "").lower() == "hotpotqa":
        compact["dataset_focus"] = "hotpotqa multi-hop evidence QA"
    return compact


def _compact_context_summary(context: Any, target: Any) -> dict[str, Any] | None:
    if not _is_hotpot_context(context):
        if context in (None, "", [], {}):
            return None
        return {"text": _truncate_text(_clean_text(context), 2000)}

    title_to_sentences = {
        str(title): sentences
        for title, sentences in context
        if isinstance(title, str) and isinstance(sentences, list)
    }
    supporting_facts = []
    if isinstance(target, dict) and isinstance(target.get("supporting_facts"), list):
        supporting_facts = target.get("supporting_facts") or []

    supporting_sentences = []
    for item in supporting_facts:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        title, sentence_index = item
        sentences = title_to_sentences.get(str(title))
        if not isinstance(sentences, list):
            continue
        try:
            index = int(sentence_index)
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(sentences):
            supporting_sentences.append(
                {
                    "title": str(title),
                    "sentence_index": index,
                    "sentence": _truncate_text(str(sentences[index]).strip(), 500),
                }
            )

    titles = list(title_to_sentences)[:20]
    return {
        "titles": titles,
        "supporting_sentences": supporting_sentences,
        "num_context_articles": len(title_to_sentences),
    }


def _is_hotpot_context(context: Any) -> bool:
    return isinstance(context, list) and all(
        isinstance(item, list)
        and len(item) == 2
        and isinstance(item[0], str)
        and isinstance(item[1], list)
        for item in context
    )


def _compact_example_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    return {
        key: metadata.get(key)
        for key in ("level", "question_type", "benchmark_format", "environment", "split")
        if key in metadata
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


def _trajectory_metadata(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    task = payload.get("task") if isinstance(payload.get("task"), dict) else {}
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    team_metadata = payload.get("team_metadata") if isinstance(payload.get("team_metadata"), dict) else {}
    return {
        "example_id": payload.get("example_id") or task.get("example_id"),
        "dataset_name": payload.get("dataset_name") or task.get("dataset_name"),
        "mas_type": payload.get("mas_type"),
        "success": _infer_success(payload),
        "metrics": metrics,
        "agent_order": team_metadata.get("agent_order"),
    }


def _infer_success(payload: Any) -> bool | None:
    if not isinstance(payload, dict):
        return None
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    success = result.get("success")
    if isinstance(success, bool):
        return success

    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    exact_match = metrics.get("exact_match")
    if isinstance(exact_match, bool):
        return exact_match
    if isinstance(exact_match, (int, float)):
        return float(exact_match) >= 1.0

    if payload.get("error") or result.get("error"):
        return False
    return None


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
    payload = _parse_payload(trajectory.content)
    profile = _prompt_profile(trajectory, payload)
    content = trajectory.content
    if max_chars > 0 and len(content) > max_chars:
        content = content[:max_chars] + "\n...[TRUNCATED]"

    system_prompt = _system_prompt(profile)
    user_prompt = f"""Trajectory id: {trajectory.trajectory_id}
Source file: {trajectory.source_file}

Extraction profile:
{json.dumps(profile, ensure_ascii=False, indent=2)}

Trajectory:
{content}
"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _parse_payload(content: str) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def _prompt_profile(trajectory: TrajectoryInput, payload: Any) -> dict[str, Any]:
    payload_metadata = _trajectory_metadata(payload)
    metadata = {**payload_metadata, **trajectory.metadata}
    dataset_name = _clean_text(metadata.get("dataset_name") or "")
    success = metadata.get("success")
    if success not in (True, False):
        success = None
    messages = payload.get("messages") if isinstance(payload, dict) else []
    team_metadata = payload.get("team_metadata") if isinstance(payload, dict) and isinstance(payload.get("team_metadata"), dict) else {}
    agent_order = metadata.get("agent_order") or team_metadata.get("agent_order") or _agent_order_from_messages(messages)
    result = payload.get("result") if isinstance(payload, dict) and isinstance(payload.get("result"), dict) else {}
    task = payload.get("task") if isinstance(payload, dict) and isinstance(payload.get("task"), dict) else {}
    task_metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    return {
        "dataset_name": dataset_name or None,
        "task_type": task.get("task_type"),
        "example_id": metadata.get("example_id"),
        "mas_type": metadata.get("mas_type"),
        "outcome": "success" if success is True else "failure" if success is False else "unknown",
        "metrics": metadata.get("metrics") or result.get("metrics") or {},
        "prediction": result.get("prediction"),
        "target": result.get("target"),
        "question_type": task_metadata.get("question_type"),
        "hotpotqa": dataset_name.lower() == "hotpotqa",
        "multi_agent": len(agent_order) > 1 or bool(team_metadata.get("topology")),
        "agent_order": agent_order,
        "topology": team_metadata.get("topology"),
        "edges": team_metadata.get("edges"),
    }


def _agent_order_from_messages(messages: Any) -> list[str]:
    order: list[str] = []
    if not isinstance(messages, list):
        return order
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = _clean_text(message.get("role") or message.get("agent_name"))
        if role and role not in order:
            order.append(role)
    return order


def _system_prompt(profile: dict[str, Any]) -> str:
    outcome = profile.get("outcome")
    if outcome == "success":
        outcome_rules = """Successful trajectory focus:
- Distill process rules that helped the agents reach the correct answer.
- Extract verification, decomposition, aggregation, or format checks that should transfer to future tasks.
- If the trajectory succeeded while exposing a process weakness, record both the success and the weakness in evidence."""
    elif outcome == "failure":
        outcome_rules = """Failed trajectory focus:
- Compare the prediction, final answer, gold target, metrics, and agent messages to isolate what went wrong.
- Extract prevention rules that would help future agents avoid the same reasoning, evidence, coordination, or answer-format failure.
- Prefer concrete warnings over generic advice; state what should be checked or changed."""
    else:
        outcome_rules = """Unknown-outcome trajectory focus:
- Extract only lessons supported directly by messages, corrections, or observable coordination patterns.
- Do not invent success or failure causes when the outcome is absent."""

    hotpotqa_rules = ""
    if profile.get("hotpotqa"):
        hotpotqa_rules = """
HotpotQA-specific extraction focus:
- Look for bridge-entity mistakes where an agent stops at the intermediate entity instead of the requested property.
- Look for comparison mistakes, answer-type mismatch, unsupported yes/no shortcuts, and wrong answer granularity.
- Use supporting evidence alignment as process evidence, but do not store source-task facts or final answers as reusable memory.
- Prefer experiences about connecting evidence pieces, checking the asked attribute, and preserving the minimal supported answer span."""

    multi_agent_rules = ""
    if profile.get("multi_agent"):
        multi_agent_rules = """
Multi-agent extraction focus:
- Preserve provenance: use a concrete producer_agent/producer_role for one agent's behavior, or multi-agent/team for coordination failures and successes.
- Prefer lessons about actor reasoning, critic verification, summarizer arbitration, handoff quality, consensus failure, and role responsibility.
- Do not decide future recipients here; allocation belongs to the downstream memory router."""

    return f"""You extract ReasoningBank-style reusable experience memories from task trajectories.

Return only a valid JSON array. Do not wrap it in Markdown.
Each array item must follow this exact schema:
{{
  "source": {{
    "producer_agent": "agent name or multi-agent or unknown",
    "producer_role": "agent role or team or unknown"
  }},
  "source_task_description": "The source task, subtask, or local situation where this experience was observed.",
  "condition": "When this experience may be applicable in future tasks.",
  "experience": "A reusable lesson, strategy, warning, or heuristic distilled from the trajectory.",
  "evidence": "The key source behavior, outcome, failure, correction, or feedback that supports this experience."
}}

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
{outcome_rules}{hotpotqa_rules}{multi_agent_rules}
"""


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

    source_payload = {
        "trajectory_id": trajectory.trajectory_id,
        "producer_agent": producer_agent,
        "producer_role": producer_role,
    }
    source_payload.update(_source_metadata(trajectory))

    sanitized = {
        "memory_id": f"m_{memory_index:06d}",
        "source": source_payload,
        "source_task_description": _clean_text(item.get("source_task_description")),
        "condition": _clean_text(item.get("condition")),
        "experience": _clean_text(item.get("experience")),
        "evidence": _clean_text(item.get("evidence")),
    }

    if any(not sanitized[field] for field in REQUIRED_OUTPUT_FIELDS):
        return None
    return sanitized


def _source_metadata(trajectory: TrajectoryInput) -> dict[str, Any]:
    payload = _parse_payload(trajectory.content)
    metadata = {**_trajectory_metadata(payload), **trajectory.metadata}
    source_metadata = {
        "example_id": metadata.get("example_id"),
        "dataset_name": metadata.get("dataset_name"),
        "mas_type": metadata.get("mas_type"),
    }
    if metadata.get("success") in (True, False):
        source_metadata["success"] = metadata.get("success")
    metrics = metadata.get("metrics")
    if isinstance(metrics, dict) and metrics:
        source_metadata["metrics"] = metrics
    return {key: value for key, value in source_metadata.items() if value not in (None, "", [], {})}


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...[truncated]"


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
