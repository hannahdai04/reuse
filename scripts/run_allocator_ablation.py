"""Run one allocator ablation setting over a target set."""

from __future__ import annotations

import argparse
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


class NoOpAllocatorLLM:
    model_name = "no-op-allocator"

    def generate(self, messages: list[dict], **kwargs):
        raise RuntimeError("Allocator LLM should not be called for this setting.")


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
    allocator_llm = (
        NoOpAllocatorLLM()
        if args.setting in {"B0", "B1"}
        else create_llm(args.allocator_provider, args.allocator_model, args.allocator_timeout, args.allocator_retries, args.allocator_max_tokens)
    )

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
    }
    record = dict(record)
    record["task_result"] = task_result
    return record


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
    for name in ("examples.jsonl", "trajectories.jsonl", "results.jsonl", "errors.jsonl", "allocation_records.jsonl"):
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


if __name__ == "__main__":
    main()
