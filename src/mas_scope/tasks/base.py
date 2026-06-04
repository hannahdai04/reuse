"""Task interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from mas_scope.core.types import AgentSpec, TaskExample, Trajectory


class BaseTask(ABC):
    task_type: str

    @abstractmethod
    def build_prompt(self, example: TaskExample, agent_spec: AgentSpec, context: dict) -> str:
        ...

    @abstractmethod
    def parse_final_answer(self, trajectory: Trajectory):
        ...

    @abstractmethod
    def evaluate(self, prediction, example: TaskExample) -> dict:
        ...

    def _memory_section(self, context: dict) -> str:
        memories = context.get("memories") or []
        if not memories:
            return ""
        lines = ["Reusable experience memories:"]
        for index, memory in enumerate(memories, start=1):
            decision = memory.get("decision") or {}
            score = decision.get("score", "")
            policy = decision.get("policy", "")
            source = memory.get("source_example_id") or memory.get("source_run_id") or "unknown"
            memory_id = memory.get("memory_id") or f"memory-{index}"
            text = str(memory.get("text") or memory.get("o_src") or "").replace("\r", " ").strip()
            if len(text) > 700:
                text = text[:700] + "...[truncated]"
            lines.append(f"[MEM-{index}] id={memory_id} score={score} policy={policy} source={source}")
            lines.append(f"content: {text}")
        lines.append("Use these only when compatible. Do not violate the task output protocol.")
        return "\n".join(lines) + "\n"
