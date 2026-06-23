"""Run one allocator ablation setting over a target set."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from allocator.diagnostics import compute_diagnostics  # noqa: E402
from allocator.retrieval import load_memories  # noqa: E402
from allocator.runtime_provider import AllocatorRuntimeMemoryProvider  # noqa: E402
from mas_scope.core.registry import registry  # noqa: E402
from mas_scope.core.types import TaskExample, TaskResult  # noqa: E402
from mas_scope.evaluation.aggregate import aggregate_metrics  # noqa: E402
from mas_scope.llm.mock import MockLLM  # noqa: E402
from mas_scope.llm.openai_compatible import OpenAICompatibleLLM  # noqa: E402
from mas_scope.memory.null_memory import NullMemoryProvider  # noqa: E402
from mas_scope.tasks.formal_planning import FormalPlanningTask  # noqa: E402
from mas_scope.tasks.interactive import InteractiveTask  # noqa: E402
from mas_scope.tasks.qa import QATask  # noqa: E402


SETTINGS = {"B0", "B1", "B2", "B3", "B4", "G2", "G3"}
CACHE_VERSION = 1
ALLOCATOR_PROMPT_VERSION = 3
UPSTREAM_STAGE_SETTINGS = {
    "B2": ["B1"],
    "B3": ["B2"],
    "B4": ["B3", "B2"],
    "G3": ["G2"],
}


class NoOpAllocatorLLM:
    model_name = "no-op-allocator"

    def generate(self, messages: list[dict], **kwargs):
        raise RuntimeError("Allocator LLM should not be called for this setting.")


class PromptTraceLLM:
    """Wraps an LLM and writes every generate call to JSONL for debugging."""

    def __init__(self, llm: Any, output_path: Path, trace_type: str) -> None:
        self.llm = llm
        self.output_path = output_path
        self.trace_type = trace_type
        self.model_name = llm.model_name
        self.call_index = 0
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def generate(self, messages: list[dict], **kwargs):
        self.call_index += 1
        response = self.llm.generate(messages, **kwargs)
        _append_jsonl(
            self.output_path,
            {
                "trace_type": self.trace_type,
                "call_index": self.call_index,
                "model_name": self.model_name,
                "kwargs": _jsonable(kwargs),
                "messages": messages,
                "response": response.content,
                "usage": response.usage,
                "raw": response.raw,
            },
        )
        return response


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    if args.setting not in SETTINGS:
        raise SystemExit(f"Unknown setting {args.setting}. Expected one of {sorted(SETTINGS)}")

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_jsonl_files(output_dir)

    examples = load_targets(args)
    completed_ids = _completed_target_ids(output_dir / "results.jsonl")
    examples_to_run = [example for example in examples if args.overwrite or example.example_id not in completed_ids]

    task = create_task(examples[0].task_type if examples else "qa")
    mas = registry.get_mas(args.mas_type)()
    task_llm = create_llm(args.task_provider, args.mas_model, args.task_timeout, args.task_retries, args.task_max_tokens)
    memories = [] if args.setting == "B0" else load_memories(args.memory_file)
    cache_config_key = _stage_cache_config_key(args)
    stage_cache_records, stage_cache_source = _load_upstream_stage_cache(args, output_dir, cache_config_key)
    allocator_llm = (
        NoOpAllocatorLLM()
        if args.setting in {"B0", "B1"}
        else create_llm(args.allocator_provider, args.allocator_model, args.allocator_timeout, args.allocator_retries, args.allocator_max_tokens)
    )
    if args.save_prompt_traces:
        task_llm = PromptTraceLLM(task_llm, output_dir / "mas_prompt_trace.jsonl", "mas")
        if args.setting not in {"B0", "B1"}:
            allocator_llm = PromptTraceLLM(allocator_llm, output_dir / "allocator_prompt_trace.jsonl", "allocator")

    manifest = {
        "setting": args.setting,
        "memory_files": [str(path) for path in args.memory_file],
        "memory_source_seed_runs": [str(path) for path in args.memory_source_seed_run],
        "target_file": str(args.target_file) if args.target_file else None,
        "dataset": args.dataset,
        "data_path": str(args.data_path) if args.data_path else None,
        "split": args.split,
        "target_offset": args.offset,
        "target_limit": args.limit,
        "target_example_ids": [example.example_id for example in examples],
        "mas_type": args.mas_type,
        "mas_model": args.mas_model,
        "allocator_model": args.allocator_model,
        "retrieval_top_k": args.retrieval_top_k,
        "per_agent_memory_k": args.per_agent_memory_k,
        "global_chunk_size": args.global_chunk_size,
        "mask_allowed_per_agent_k": args.mask_allowed_per_agent_k,
        "memory_count": len(memories),
        "resume_skipped": len(examples) - len(examples_to_run),
        "stage_cache_mode": args.stage_cache_mode,
        "stage_cache_config_key": cache_config_key,
        "stage_cache_source": stage_cache_source,
        "save_prompt_traces": args.save_prompt_traces,
    }
    _write_json(output_dir / "manifest.json", manifest)

    results = _read_results(output_dir / "results.jsonl") if not args.overwrite else []
    records = _read_jsonl(output_dir / "allocation_records.jsonl") if not args.overwrite else []

    for example in examples_to_run:
        _append_jsonl(output_dir / "examples.jsonl", example)
        provider = (
            NullMemoryProvider()
            if args.setting == "B0"
            else AllocatorRuntimeMemoryProvider(
                setting=args.setting,
                memories=memories,
                allocator_llm=allocator_llm,
                retrieval_top_k=args.retrieval_top_k,
                per_agent_memory_k=args.per_agent_memory_k,
                global_chunk_size=args.global_chunk_size,
                mask_allowed_per_agent_k=args.mask_allowed_per_agent_k,
                masking_max_tokens=args.allocator_max_tokens,
                selection_max_tokens=args.allocator_max_tokens,
                realization_max_tokens=args.allocator_max_tokens,
                stage_cache_records=stage_cache_records,
                stage_cache_source=stage_cache_source,
            )
        )
        try:
            if example.task_type == "interactive":
                trajectory = task.run_episode(example, mas, task_llm, provider)
                prediction = task.parse_final_answer(trajectory)
                metrics = task.evaluate_trajectory(trajectory)
            else:
                trajectory = mas.run(example, task, task_llm, provider)
                prediction = task.parse_final_answer(trajectory)
                metrics = task.evaluate(prediction, example)
            result = TaskResult(
                run_id=trajectory.run_id,
                example_id=example.example_id,
                prediction=prediction,
                target=example.target,
                metrics=metrics,
                success=_success(metrics),
                cost=trajectory.usage,
            )
            _append_jsonl(output_dir / "trajectories.jsonl", trajectory)
        except Exception as exc:
            _append_jsonl(output_dir / "errors.jsonl", {"example_id": example.example_id, "error": str(exc)})
            result = TaskResult(
                run_id="",
                example_id=example.example_id,
                prediction=None,
                target=example.target,
                metrics={},
                success=False,
                error=str(exc),
            )

        record = _allocation_record(args, example, mas, provider, result)
        _append_jsonl(output_dir / "allocation_records.jsonl", record)
        for trace_row in record.get("agent_memory_trace") or []:
            _append_jsonl(output_dir / "agent_memory_trace.jsonl", trace_row)
        _append_jsonl(output_dir / "stage_cache.jsonl", _stage_cache_record(record, cache_config_key))
        _append_jsonl(output_dir / "results.jsonl", result)
        records.append(record)
        results.append(result)

    _write_json(output_dir / "metrics.json", aggregate_metrics(results))
    _write_json(output_dir / "diagnostics.json", compute_diagnostics(records))
    print(output_dir)
    return output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run allocator B0-B4 ablation.")
    parser.add_argument("--memory_file", action="append", default=[], type=Path)
    parser.add_argument("--memory_source_seed_run", action="append", default=[], type=Path)
    parser.add_argument("--target_file", type=Path, default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--data_path", type=Path, default=None)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--retrieval_top_k", type=int, default=10)
    parser.add_argument("--per_agent_memory_k", type=int, default=3)
    parser.add_argument("--global_chunk_size", type=int, default=20)
    parser.add_argument("--mask_allowed_per_agent_k", type=int, default=10)
    parser.add_argument("--mas_type", default="macnet")
    parser.add_argument("--mas_model", default="Qwen/Qwen3-8B")
    parser.add_argument("--task_provider", default="openai-compatible", choices=["mock", "openai-compatible"])
    parser.add_argument("--task_timeout", type=float, default=90.0)
    parser.add_argument("--task_retries", type=int, default=2)
    parser.add_argument("--task_max_tokens", type=int, default=32)
    parser.add_argument("--allocator_provider", default="openai-compatible", choices=["mock", "openai-compatible"])
    parser.add_argument("--allocator_model", default="Qwen/Qwen3-8B")
    parser.add_argument("--allocator_timeout", type=float, default=90.0)
    parser.add_argument("--allocator_retries", type=int, default=1)
    parser.add_argument("--allocator_max_tokens", type=int, default=1200)
    parser.add_argument("--stage_cache_mode", default="auto", choices=["auto", "off"])
    parser.add_argument("--save_prompt_traces", action="store_true")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def load_targets(args: argparse.Namespace) -> list[TaskExample]:
    if args.target_file:
        rows = _read_jsonl(args.target_file)
        examples = [TaskExample.model_validate(row) for row in rows]
    else:
        if not args.dataset or not args.data_path:
            raise SystemExit("Provide either --target_file or --dataset with --data_path.")
        builder = registry.get_dataset_builder(args.dataset)(args.data_path)
        build_limit = None if args.limit is None else args.limit + args.offset
        examples = builder.build(args.split, build_limit)
        if args.offset:
            examples = examples[args.offset :]
        if args.limit is not None:
            examples = examples[: args.limit]
    if not examples:
        raise SystemExit("No target examples loaded.")
    return examples


def create_task(task_type: str):
    if task_type == "qa":
        return QATask()
    if task_type == "formal_planning":
        return FormalPlanningTask()
    return InteractiveTask(None)


def create_llm(provider: str, model: str, timeout: float, retries: int, max_tokens: int | None):
    if provider == "mock":
        return MockLLM(model_name=model or "mock-llm")
    return OpenAICompatibleLLM(
        model_name=model,
        timeout=timeout,
        retries=retries,
        temperature=0,
        max_tokens=max_tokens,
        extra_body={"enable_thinking": False},
    )


def _allocation_record(args, example, mas, provider, result: TaskResult) -> dict[str, Any]:
    task_result = result.model_dump(mode="json")
    if args.setting == "B0":
        return {
            "target_task_id": example.example_id,
            "setting": args.setting,
            "retrieval_top_k": args.retrieval_top_k,
            "per_agent_memory_k": args.per_agent_memory_k,
            "global_chunk_size": args.global_chunk_size,
            "mask_allowed_per_agent_k": args.mask_allowed_per_agent_k,
            "global_masking": False,
            "allowed_cap_stats": {},
            "retrieved_memories": [],
            "agents": [agent.role for agent in mas.agents_list],
            "mask_matrix": [],
            "mask_reasons": [],
            "selected_matrix": [],
            "realized_memories": {},
            "task_result": task_result,
            "allocator_errors": [],
            "cache_metadata": {
                "stage_cache_mode": args.stage_cache_mode,
                "cache_hit": False,
                "source_setting": None,
                "source_path": None,
                "source_cache_status": None,
                "reused_stages": [],
            },
            "agent_memory_trace": [],
        }
    record = provider.get_record(example.example_id) or {
        "target_task_id": example.example_id,
        "setting": args.setting,
        "retrieved_memories": [],
        "agents": [agent.role for agent in mas.agents_list],
        "mask_matrix": [],
        "mask_reasons": [],
        "selected_matrix": [],
        "realized_memories": {},
        "allocator_errors": ["allocator_record_missing"],
        "cache_metadata": {
            "stage_cache_mode": args.stage_cache_mode,
            "cache_hit": False,
            "source_setting": None,
            "source_path": None,
            "source_cache_status": None,
            "reused_stages": [],
        },
    }
    record = dict(record)
    record["task_result"] = task_result
    record["agent_memory_trace"] = _agent_memory_trace(record)
    return record


def _agent_memory_trace(record: dict[str, Any]) -> list[dict[str, Any]]:
    target_task_id = str(record.get("target_task_id") or "")
    setting = str(record.get("setting") or "")
    agents = [str(agent) for agent in record.get("agents") or []]
    memories = record.get("retrieved_memories") or []
    mask_matrix = record.get("mask_matrix") or []
    selected_matrix = record.get("selected_matrix") or []
    realized = record.get("realized_memories") or {}
    task_result = record.get("task_result") or {}
    reason_by_memory = {
        str(item.get("memory_id")): str(item.get("reason_tag") or "")
        for item in record.get("mask_reasons") or []
    }
    realized_by_pair = {
        (str(agent), str(item.get("memory_id"))): item
        for agent, items in realized.items()
        for item in (items or [])
    }

    rows: list[dict[str, Any]] = []
    for memory_index, memory in enumerate(memories):
        memory_id = str(memory.get("memory_id") or "")
        for agent_index, agent in enumerate(agents):
            mask_allowed = _matrix_value(mask_matrix, memory_index, agent_index)
            selected = _matrix_value(selected_matrix, memory_index, agent_index)
            realization = realized_by_pair.get((agent, memory_id))
            injected = realization is not None
            if injected:
                stage_status = "injected"
            elif selected:
                stage_status = "selected_missing_realization"
            elif mask_allowed:
                stage_status = "allowed_not_selected"
            else:
                stage_status = "rejected_by_mask"
            rows.append(
                {
                    "target_task_id": target_task_id,
                    "setting": setting,
                    "agent": agent,
                    "agent_index": agent_index,
                    "memory_id": memory_id,
                    "original_memory_id": memory.get("original_memory_id"),
                    "memory_namespace": memory.get("memory_namespace"),
                    "retrieval_rank": memory.get("retrieval_rank"),
                    "retrieval_score": memory.get("retrieval_score"),
                    "mask_allowed": bool(mask_allowed),
                    "selected": bool(selected),
                    "injected": bool(injected),
                    "stage_status": stage_status,
                    "mask_reason_tag": reason_by_memory.get(memory_id, ""),
                    "realization_type": realization.get("realization_type") if realization else None,
                    "realized_text": realization.get("text") if realization else None,
                    "realized_text_length": len(str(realization.get("text") or "")) if realization else 0,
                    "memory_condition": _short(memory.get("condition"), 360),
                    "memory_experience": _short(memory.get("experience"), 520),
                    "memory_text_preview": _short(memory.get("text"), 520),
                    "task_success": bool(task_result.get("success")),
                    "task_score": _task_score(task_result),
                    "task_prediction": task_result.get("prediction"),
                    "task_target": task_result.get("target"),
                    "cache_hit": bool((record.get("cache_metadata") or {}).get("cache_hit")),
                    "cache_source_setting": (record.get("cache_metadata") or {}).get("source_setting"),
                    "allocator_error_count": len(record.get("allocator_errors") or []),
                }
            )
    return rows


def _matrix_value(matrix: list[list[Any]], row: int, col: int) -> int:
    if row >= len(matrix) or col >= len(matrix[row]):
        return 0
    try:
        return 1 if int(matrix[row][col]) else 0
    except (TypeError, ValueError):
        return 0


def _short(value: Any, limit: int) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _task_score(result: dict[str, Any]) -> float:
    metrics = result.get("metrics") or {}
    for key in ("exact_match", "yes_no_accuracy", "success", "valid_action_format"):
        if key in metrics and isinstance(metrics[key], (int, float, bool)):
            return float(metrics[key])
    numeric = [float(value) for value in metrics.values() if isinstance(value, (int, float, bool))]
    if numeric:
        return sum(numeric) / len(numeric)
    return 1.0 if result.get("success") else 0.0


def _success(metrics: dict) -> bool:
    if "success" in metrics:
        return bool(metrics["success"])
    if metrics.get("scored") is False:
        return True
    if "exact_match" in metrics:
        return bool(metrics["exact_match"])
    if "valid_action_format" in metrics:
        return bool(metrics["valid_action_format"])
    return False


def _ensure_jsonl_files(output_dir: Path) -> None:
    for name in (
        "examples.jsonl",
        "trajectories.jsonl",
        "results.jsonl",
        "errors.jsonl",
        "allocation_records.jsonl",
        "agent_memory_trace.jsonl",
        "stage_cache.jsonl",
    ):
        path = output_dir / name
        if not path.exists():
            path.write_text("", encoding="utf-8")


def _completed_target_ids(results_path: Path) -> set[str]:
    return {str(row.get("example_id")) for row in _read_jsonl(results_path) if row.get("example_id")}


def _read_results(path: Path) -> list[TaskResult]:
    return [TaskResult.model_validate(row) for row in _read_jsonl(path)]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, payload: Any) -> None:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _stage_cache_config_key(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "memory_files_hash": _memory_files_hash(args.memory_file),
        "mas_type": args.mas_type,
        "retrieval_top_k": args.retrieval_top_k,
        "global_chunk_size": args.global_chunk_size,
        "mask_allowed_per_agent_k": args.mask_allowed_per_agent_k,
        "per_agent_memory_k": args.per_agent_memory_k,
        "candidate_mode": "global" if args.setting in {"G2", "G3"} else "retrieval",
        "allocator_prompt_version": ALLOCATOR_PROMPT_VERSION,
    }


def _memory_files_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        digest.update(str(path).replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        if path.exists():
            digest.update(path.read_bytes())
        else:
            digest.update(b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()


def _load_upstream_stage_cache(
    args: argparse.Namespace,
    output_dir: Path,
    config_key: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    source = {
        "mode": args.stage_cache_mode,
        "source_setting": None,
        "source_path": None,
        "loaded_count": 0,
        "rejected_count": 0,
    }
    if args.stage_cache_mode == "off":
        return {}, source

    for upstream_setting in UPSTREAM_STAGE_SETTINGS.get(args.setting, []):
        path = output_dir.parent / upstream_setting / "stage_cache.jsonl"
        if not path.exists():
            continue
        rows = _read_jsonl(path)
        valid: dict[str, dict[str, Any]] = {}
        rejected = 0
        for row in rows:
            if row.get("cache_version") != CACHE_VERSION or row.get("config_key") != config_key:
                rejected += 1
                continue
            target_id = str(row.get("target_task_id") or "")
            if target_id:
                valid[target_id] = row
        if valid:
            source.update(
                {
                    "source_setting": upstream_setting,
                    "source_path": str(path),
                    "loaded_count": len(valid),
                    "rejected_count": rejected,
                }
            )
            return valid, source
    return {}, source


def _stage_cache_record(record: dict[str, Any], config_key: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_task_id": record.get("target_task_id"),
        "setting": record.get("setting"),
        "cache_version": CACHE_VERSION,
        "config_key": config_key,
        "agents": record.get("agents") or [],
        "retrieved_memories": record.get("retrieved_memories") or [],
        "mask_matrix": record.get("mask_matrix") or [],
        "selected_matrix": record.get("selected_matrix") or [],
        "cache_status": "allocator_error" if record.get("allocator_errors") else "ok",
    }


if __name__ == "__main__":
    main()
