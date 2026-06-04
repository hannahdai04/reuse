"""Base MAS interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from mas_scope.core.ids import make_run_id
from mas_scope.core.types import ActionDecision, AgentSpec, MessageGraph, TaskExample, Trajectory
from mas_scope.llm.base import BaseLLM
from mas_scope.mas.agent import MASAgent
from mas_scope.memory.base import MemoryProvider


class BaseMAS(ABC):
    mas_type: str
    agents_list: list[MASAgent]
    topology: dict[str, list[str]]

    @abstractmethod
    def run(self, example: TaskExample, task, llm: BaseLLM, memory: MemoryProvider) -> Trajectory:
        ...

    @abstractmethod
    def act(
        self,
        example: TaskExample,
        observation: str,
        task,
        llm: BaseLLM,
        memory: MemoryProvider,
        context: dict,
    ) -> ActionDecision:
        ...

    def _agent_spec(self, agent: MASAgent) -> AgentSpec:
        return AgentSpec(
            name=agent.name,
            role=agent.role,
            system_prompt=agent.system_prompt_template,
        )

    def _task_description(self, example: TaskExample, task, agent: MASAgent, context: dict | None = None) -> str:
        if hasattr(task, "build_prompt"):
            return task.build_prompt(example, self._agent_spec(agent), context or {})
        return str(example.input)

    def _system_inputs(self, example: TaskExample) -> dict:
        return {
            "task_type": example.task_type,
            "task_domain_instructions": example.task_type,
        }

    def _llm_kwargs(self, example: TaskExample, observation: str = "") -> dict:
        return {
            "task_type": example.task_type,
            "target": example.target,
            "observation": observation,
        }

    def _task_description_with_memory(
        self,
        example: TaskExample,
        task,
        agent: MASAgent,
        context: dict,
        memory: MemoryProvider,
    ) -> tuple[str, dict]:
        memory_context = dict(context)
        memory_context["agent_order"] = [item.role for item in self.agents_list]
        memory_context["current_agent"] = agent.role
        try:
            selected_memories = memory.retrieve(example, self._agent_spec(agent), memory_context)
            provider_metadata = getattr(memory, "last_retrieval_metadata", {}) or {}
        except Exception as exc:
            selected_memories = []
            provider_metadata = {
                "candidate_count": 0,
                "selected_ids": [],
                "agent_order": memory_context["agent_order"],
                "current_agent": agent.role,
                "agent_mask_rows": [],
                "decisions": [],
                "policy_mode": "error",
                "errors": [str(exc)],
            }
        prompt_context = {
            **context,
            "memories": selected_memories,
            "memory_metadata": provider_metadata,
        }
        return self._task_description(example, task, agent, prompt_context), provider_metadata

    def _retrieve_memory(self, memory: MemoryProvider, example: TaskExample, agent: MASAgent, context: dict) -> tuple[list[dict], dict]:
        memory_context = dict(context)
        memory_context["agent_order"] = [item.role for item in self.agents_list]
        memory_context["current_agent"] = agent.role
        selected_memories = memory.retrieve(example, self._agent_spec(agent), memory_context)
        return selected_memories, getattr(memory, "last_retrieval_metadata", {}) or {}

    def _trajectory_from_graph(
        self,
        example: TaskExample,
        llm: BaseLLM,
        graph: MessageGraph,
        metadata: dict,
    ) -> Trajectory:
        return Trajectory(
            run_id=make_run_id("trajectory"),
            example_id=example.example_id,
            dataset_name=example.dataset_name,
            mas_type=self.mas_type,
            model_name=llm.model_name,
            messages=graph.messages,
            final_answer=graph.action,
            usage=self._usage_from_messages(graph.messages),
            metadata=metadata,
        )

    def _usage_from_messages(self, messages) -> dict:
        usage: dict[str, float] = {}
        for message in messages:
            message_usage = message.metadata.get("llm_usage", {})
            for key, value in message_usage.items():
                if isinstance(value, (int, float)):
                    usage[key] = usage.get(key, 0) + value
        return usage
