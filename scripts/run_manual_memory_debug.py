"""Run a single MAS task with manually routed memories for debugging."""

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

import mas_scope.mas  # noqa: F401,E402 - registers MAS classes
from allocator.retrieval import load_memories  # noqa: E402
from mas_scope.core.registry import registry  # noqa: E402
from mas_scope.core.types import AgentSpec, TaskExample, TaskResult  # noqa: E402
from mas_scope.llm.mock import MockLLM  # noqa: E402
from mas_scope.llm.openai_compatible import OpenAICompatibleLLM  # noqa: E402
from mas_scope.memory.base import MemoryProvider  # noqa: E402
from mas_scope.memory.null_memory import NullMemoryProvider  # noqa: E402
from mas_scope.tasks.formal_planning import FormalPlanningTask  # noqa: E402
from mas_scope.tasks.interactive import InteractiveTask  # noqa: E402
from mas_scope.tasks.qa import QATask  # noqa: E402


class PromptTraceLLM:
    """Wrap an LLM and save every prompt/response pair."""

    def __init__(self, llm: Any, output_path: Path) -> None:
        self.llm = llm
        self.output_path = output_path
        self.model_name = llm.model_name
        self.call_index = 0
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def generate(self, messages: list[dict], **kwargs):
        self.call_index += 1
        response = self.llm.generate(messages, **kwargs)
        append_jsonl(
            self.output_path,
            {
                "call_index": self.call_index,
                "model_name": self.model_name,
                "kwargs": jsonable(kwargs),
                "messages": messages,
                "response": response.content,
                "usage": response.usage,
                "raw": response.raw,
            },
        )
        return response


class ManualMemoryProvider(MemoryProvider):
    """Serve a fixed agent->memory mapping through the normal MAS memory hook."""

    def __init__(self, memories: list[dict[str, Any]], mapping: dict[str, list[str]]) -> None:
        self.memory_by_id = _memory_lookup(memories)
        self.mapping = mapping
        self.manual_ids = _ordered_unique(memory_id for ids in mapping.values() for memory_id in ids)
        self.last_retrieval_metadata: dict[str, Any] = {}
        self.trace_rows: list[dict[str, Any]] = []
        missing = [memory_id for memory_id in self.manual_ids if memory_id not in self.memory_by_id]
        if missing:
            raise ValueError(f"Manual memory ids not found: {missing}")

    def retrieve(self, example: TaskExample, agent_spec: AgentSpec, context: dict) -> list[dict]:
        agents = [str(agent) for agent in context.get("agent_order") or [agent_spec.role]]
        current_agent = str(context.get("current_agent") or agent_spec.role)
        selected_ids = [self._canonical(memory_id) for memory_id in self.mapping.get(current_agent, [])]
        selected = [self._injectable_memory(self.memory_by_id[memory_id]) for memory_id in selected_ids]
        mask_rows = self._mask_rows(agents)
        self.last_retrieval_metadata = {
            "candidate_count": len(self.manual_ids),
            "selected_ids": selected_ids,
            "agent_order": agents,
            "current_agent": current_agent,
            "agent_mask_rows": mask_rows,
            "decisions": [
                {
                    "memory_id": memory_id,
                    "score": "manual",
                    "agent_mask": mask_rows[index],
                    "policy": "manual",
                    "selected_for_current_agent": memory_id in selected_ids,
                }
                for index, memory_id in enumerate(self.manual_ids)
            ],
            "policy_mode": "manual",
            "errors": [],
        }
        for memory_id in self.manual_ids:
            memory = self.memory_by_id[memory_id]
            self.trace_rows.append(
                {
                    "target_task_id": example.example_id,
                    "agent": current_agent,
                    "memory_id": memory_id,
                    "selected": memory_id in selected_ids,
                    "injected": memory_id in selected_ids,
                    "memory_condition": memory.get("condition"),
                    "memory_experience": memory.get("experience"),
                    "memory_evidence": memory.get("evidence"),
                    "memory_text": memory.get("text"),
                }
            )
        return selected

    def update(self, example: TaskExample, trajectory, result) -> None:
        return None

    def _canonical(self, memory_id: str) -> str:
        if memory_id in self.memory_by_id:
            return memory_id
        candidates = [key for key, memory in self.memory_by_id.items() if memory.get("original_memory_id") == memory_id]
        if len(candidates) == 1:
            return candidates[0]
        return memory_id

    def _mask_rows(self, agents: list[str]) -> list[list[int]]:
        rows: list[list[int]] = []
        canonical_mapping = {
            agent: {self._canonical(memory_id) for memory_id in memory_ids}
            for agent, memory_ids in self.mapping.items()
        }
        for memory_id in self.manual_ids:
            rows.append([1 if memory_id in canonical_mapping.get(agent, set()) else 0 for agent in agents])
        return rows

    def _injectable_memory(self, memory: dict[str, Any]) -> dict[str, Any]:
        return {
            "memory_id": memory.get("memory_id"),
            "original_memory_id": memory.get("original_memory_id"),
            "memory_namespace": memory.get("memory_namespace"),
            "text": memory.get("text"),
            "source_example_id": memory.get("memory_namespace"),
            "decision": {
                "score": "manual",
                "policy": "manual",
                "realization_type": "manual_raw",
            },
        }


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_targets(args)
    if len(examples) != 1:
        raise SystemExit(f"Manual debug expects exactly one target; loaded {len(examples)}.")
    example = examples[0]
    task = create_task(example.task_type)
    mas = registry.get_mas(args.mas_type)()
    task_llm = create_llm(args.task_provider, args.mas_model, args.task_timeout, args.task_retries, args.task_max_tokens)
    if args.save_prompt_traces:
        task_llm = PromptTraceLLM(task_llm, output_dir / "mas_prompt_trace.jsonl")

    mapping = parse_mapping(args.manual_memory)
    memories = []
    if mapping:
        memories.extend(load_memories(args.memory_file))
        oracle_memories = parse_oracle_memories(args.oracle_memory)
        memories.extend(oracle_memories.values())
    provider: MemoryProvider
    if mapping:
        provider = ManualMemoryProvider(memories, mapping)
    else:
        provider = NullMemoryProvider()

    manifest = {
        "mode": "manual_memory_debug",
        "memory_files": [str(path) for path in args.memory_file],
        "oracle_memory_ids": list(parse_oracle_memories(args.oracle_memory).keys()),
        "manual_mapping": mapping,
        "dataset": args.dataset,
        "data_path": str(args.data_path) if args.data_path else None,
        "split": args.split,
        "offset": args.offset,
        "limit": args.limit,
        "example_id": args.example_id,
        "target_example_id": example.example_id,
        "mas_type": args.mas_type,
        "mas_model": args.mas_model,
        "save_prompt_traces": args.save_prompt_traces,
    }
    write_json(output_dir / "manifest.json", manifest)
    append_jsonl(output_dir / "examples.jsonl", example)

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
            success=success(metrics),
            cost=trajectory.usage,
        )
        append_jsonl(output_dir / "trajectories.jsonl", trajectory)
    except Exception as exc:
        append_jsonl(output_dir / "errors.jsonl", {"example_id": example.example_id, "error": str(exc)})
        result = TaskResult(
            run_id="",
            example_id=example.example_id,
            prediction=None,
            target=example.target,
            metrics={},
            success=False,
            error=str(exc),
        )

    append_jsonl(output_dir / "results.jsonl", result)
    if isinstance(provider, ManualMemoryProvider):
        for row in provider.trace_rows:
            append_jsonl(output_dir / "agent_memory_trace.jsonl", row)
    write_report(output_dir / "summary.md", example, result, provider)
    print(output_dir)
    return output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug one task with manually injected memories.")
    parser.add_argument("--memory_file", action="append", default=[], type=Path)
    parser.add_argument("--oracle_memory", action="append", default=[], help='Format: "memory_id=memory text"')
    parser.add_argument("--manual_memory", action="append", default=[], help='Format: "agent role=memory_id,memory_id"')
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--data_path", type=Path, default=None)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--example_id", default=None)
    parser.add_argument("--mas_type", default="macnet")
    parser.add_argument("--mas_model", default="Qwen/Qwen3-8B")
    parser.add_argument("--task_provider", default="openai-compatible", choices=["mock", "openai-compatible"])
    parser.add_argument("--task_timeout", type=float, default=90.0)
    parser.add_argument("--task_retries", type=int, default=2)
    parser.add_argument("--task_max_tokens", type=int, default=32)
    parser.add_argument("--save_prompt_traces", action="store_true")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def parse_mapping(items: list[str]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Invalid --manual_memory value: {item!r}. Expected 'agent role=m_000001,m_000002'.")
        agent, memory_ids = item.split("=", 1)
        agent = agent.strip()
        ids = [memory_id.strip() for memory_id in memory_ids.split(",") if memory_id.strip()]
        if not agent or not ids:
            raise SystemExit(f"Invalid --manual_memory value: {item!r}.")
        mapping.setdefault(agent, []).extend(ids)
    return {agent: _ordered_unique(ids) for agent, ids in mapping.items()}


def parse_oracle_memories(items: list[str]) -> dict[str, dict[str, Any]]:
    memories: dict[str, dict[str, Any]] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Invalid --oracle_memory value: {item!r}. Expected 'memory_id=memory text'.")
        memory_id, text = item.split("=", 1)
        memory_id = memory_id.strip()
        text = text.strip()
        if not memory_id or not text:
            raise SystemExit(f"Invalid --oracle_memory value: {item!r}.")
        memories[memory_id] = {
            "memory_id": memory_id,
            "original_memory_id": memory_id,
            "memory_namespace": "oracle",
            "source": {
                "trajectory_id": "manual_oracle",
                "producer_agent": "human",
                "producer_role": "oracle",
            },
            "source_task_description": "Human-authored oracle memory for bridge-entity answer-type debugging.",
            "condition": "When debugging whether manually selected memories can correct bridge-entity-as-final-answer failures.",
            "experience": text,
            "evidence": "Human-authored oracle memory for a controlled single-task intervention.",
            "text": text,
        }
    return memories


def load_targets(args: argparse.Namespace) -> list[TaskExample]:
    if not args.dataset or not args.data_path:
        raise SystemExit("Provide --dataset and --data_path.")
    builder = registry.get_dataset_builder(args.dataset)(args.data_path)
    build_limit = None if args.limit is None else args.limit + args.offset
    examples = builder.build(args.split, build_limit)
    if args.example_id:
        examples = [example for example in examples if example.example_id == args.example_id]
    else:
        if args.offset:
            examples = examples[args.offset :]
        if args.limit is not None:
            examples = examples[: args.limit]
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


def success(metrics: dict) -> bool:
    if "success" in metrics:
        return bool(metrics["success"])
    if metrics.get("scored") is False:
        return True
    if "exact_match" in metrics:
        return bool(metrics["exact_match"])
    if "valid_action_format" in metrics:
        return bool(metrics["valid_action_format"])
    return False


def write_report(path: Path, example: TaskExample, result: TaskResult, provider: MemoryProvider) -> None:
    lines = [
        "# Manual Memory Debug Summary",
        "",
        f"- example_id: `{example.example_id}`",
        f"- question: {example.input.get('question')}",
        f"- prediction: `{result.prediction}`",
        f"- target: `{example.target.get('answer')}`",
        f"- success: `{result.success}`",
        f"- metrics: `{json.dumps(result.metrics, ensure_ascii=False)}`",
        "",
    ]
    if isinstance(provider, ManualMemoryProvider):
        lines.append("## Manual Mapping")
        for agent, memory_ids in provider.mapping.items():
            lines.append(f"- {agent}: {', '.join(memory_ids)}")
        lines.append("")
        lines.append("## Injected Memory Text")
        for agent, memory_ids in provider.mapping.items():
            lines.append(f"### {agent}")
            for memory_id in memory_ids:
                canonical = provider._canonical(memory_id)
                memory = provider.memory_by_id[canonical]
                lines.append(f"- `{canonical}`: {memory.get('experience')}")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _memory_lookup(memories: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    bare: dict[str, list[str]] = {}
    for memory in memories:
        memory_id = str(memory.get("memory_id") or "")
        original_id = str(memory.get("original_memory_id") or "")
        lookup[memory_id] = memory
        if memory_id.startswith("oracle_"):
            lookup[original_id] = memory
        bare.setdefault(original_id, []).append(memory_id)
    for original_id, namespaced_ids in bare.items():
        hotpot_ids = [memory_id for memory_id in namespaced_ids if memory_id.startswith("hotpotqa:")]
        if len(hotpot_ids) == 1:
            lookup[original_id] = lookup[hotpot_ids[0]]
        elif len(namespaced_ids) == 1:
            lookup[original_id] = lookup[namespaced_ids[0]]
    return lookup


def _ordered_unique(items) -> list[str]:
    seen = set()
    unique = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def append_jsonl(path: Path, payload: Any) -> None:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


if __name__ == "__main__":
    main()
