"""In-house CAMEL-style conversational MAS."""

from __future__ import annotations

from mas_scope.core.registry import registry
from mas_scope.core.types import ActionDecision, MessageGraph, TaskExample, Trajectory
from mas_scope.llm.base import BaseLLM
from mas_scope.mas.agent import MASAgent
from mas_scope.mas.base import BaseMAS
from mas_scope.memory.base import MemoryProvider
from mas_scope.prompts.templates import (
    CAMEL_ASSISTANT_SYSTEM_PROMPT_TEMPLATE,
    CAMEL_ASSISTANT_USER_PROMPT_TEMPLATE,
    CAMEL_FINALIZER_SYSTEM_PROMPT_TEMPLATE,
    CAMEL_FINALIZER_USER_PROMPT_TEMPLATE,
    CAMEL_USER_SYSTEM_PROMPT_TEMPLATE,
    CAMEL_USER_USER_PROMPT_TEMPLATE,
)
from mas_scope.tools.action_parser import parse_action


@registry.register_mas("camel")
class CAMELStyleMAS(BaseMAS):
    mas_type = "camel"

    def __init__(self, mas_config: dict | None = None) -> None:
        self.mas_config = mas_config or {}
        self.assistant_agent = MASAgent("assistant agent", 0, CAMEL_ASSISTANT_SYSTEM_PROMPT_TEMPLATE, CAMEL_ASSISTANT_USER_PROMPT_TEMPLATE)
        self.user_agent = MASAgent("user agent", 1, CAMEL_USER_SYSTEM_PROMPT_TEMPLATE, CAMEL_USER_USER_PROMPT_TEMPLATE)
        self.finalizer_agent = MASAgent("finalizer agent", 2, CAMEL_FINALIZER_SYSTEM_PROMPT_TEMPLATE, CAMEL_FINALIZER_USER_PROMPT_TEMPLATE)
        self.agents_list = [self.assistant_agent, self.user_agent, self.finalizer_agent]
        self.topology = {
            "assistant agent": [],
            "user agent": ["assistant agent"],
            "finalizer agent": ["assistant agent", "user agent"],
        }

    def run(self, example: TaskExample, task, llm: BaseLLM, memory: MemoryProvider) -> Trajectory:
        graph = self._generate_graph(example, task, llm, memory, observation="")
        return self._trajectory_from_graph(example, llm, graph, self._metadata(graph))

    def act(self, example: TaskExample, observation: str, task, llm: BaseLLM, memory: MemoryProvider, context: dict) -> ActionDecision:
        graph = self._generate_graph(example, task, llm, memory, observation=observation, context=context)
        graph.action = parse_action(str(graph.action or ""))
        return ActionDecision(action=str(graph.action), messages=graph.messages, metadata=self._metadata(graph))

    def _generate_graph(self, example, task, llm, memory, observation: str, context: dict | None = None) -> MessageGraph:
        graph = MessageGraph(state=example.input)
        system_inputs = self._system_inputs(example)

        assistant_msg = self._invoke(graph, example, task, memory, self.assistant_agent, llm, system_inputs, {
            "dialogue_context": f"Observation: {observation}",
        }, 0, None)
        user_msg = self._invoke(graph, example, task, memory, self.user_agent, llm, system_inputs, {
            "assistant_output": assistant_msg.content,
        }, 1, [self.assistant_agent.role])
        dialogue_context = f"{assistant_msg.role}: {assistant_msg.content}\n{user_msg.role}: {user_msg.content}"
        assistant_msg_2 = self._invoke(graph, example, task, memory, self.assistant_agent, llm, system_inputs, {
            "dialogue_context": dialogue_context,
        }, 2, [self.user_agent.role])
        final_context = f"{dialogue_context}\n{assistant_msg_2.role}: {assistant_msg_2.content}"
        final_msg = self._invoke(graph, example, task, memory, self.finalizer_agent, llm, system_inputs, {
            "dialogue_context": final_context,
        }, 3, [self.assistant_agent.role, self.user_agent.role])
        graph.action = final_msg.content
        return graph

    def _invoke(self, graph, example, task, memory, agent, llm, system_inputs, user_inputs, turn_id, parent_roles):
        task_description, memory_metadata = self._task_description_with_memory(
            example,
            task,
            agent,
            {"graph": graph.model_dump(mode="json"), **user_inputs},
            memory,
        )
        msg = agent.invoke(
            llm,
            system_inputs,
            {"task_description": task_description, **user_inputs},
            turn_id,
            parent_roles,
            metadata={"memory": memory_metadata},
            llm_kwargs=self._llm_kwargs(example, user_inputs.get("dialogue_context", "")),
        )
        graph.update_message_graph(msg, agent.role, parent_roles)
        return msg

    def _metadata(self, graph: MessageGraph) -> dict:
        return {"mas_style": "camel", "topology": self.topology, "edges": graph.edges}
