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
        metadata = context.get("memory_metadata") or {}
        current_agent = str(metadata.get("current_agent") or "").lower()
        if "critic" in current_agent or "verifier" in current_agent:
            title = "Critic Verification Checklist"
        elif "summarizer" in current_agent or "final" in current_agent or "proxy" in current_agent:
            title = "Final Adjudication Checklist"
        elif "actor" in current_agent or "reasoner" in current_agent or "assistant" in current_agent:
            title = "Actor Operating Rules"
        else:
            title = "Reusable Operating Rules"
        lines = [
            title + ":",
            "Use these memories as process guidance for this agent role. They are not evidence or facts for the current task.",
        ]
        for index, memory in enumerate(memories, start=1):
            text = str(memory.get("text") or memory.get("o_src") or "").replace("\r", " ").strip()
            if len(text) > 700:
                text = text[:700] + "...[truncated]"
            lines.append(f"M{index}. {text}")
        lines.append("Apply compatible rules while following Target Evidence, Task Requirements, and the Output Contract.")
        return "\n".join(lines) + "\n"
