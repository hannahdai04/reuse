"""In-house AutoGen-style fixed topology MAS."""

from __future__ import annotations

from mas_scope.core.registry import registry
from mas_scope.core.types import ActionDecision, MessageGraph, TaskExample, Trajectory
from mas_scope.llm.base import BaseLLM
from mas_scope.mas.agent import MASAgent
from mas_scope.mas.base import BaseMAS
from mas_scope.memory.base import MemoryProvider
from mas_scope.prompts.templates import (
    AUTOGEN_ASSISTANT_SYSTEM_PROMPT_TEMPLATE,
    AUTOGEN_ASSISTANT_USER_PROMPT_TEMPLATE,
    AUTOGEN_USER_PROXY_SYSTEM_PROMPT_TEMPLATE,
    AUTOGEN_USER_PROXY_USER_PROMPT_TEMPLATE,
)
from mas_scope.tools.action_parser import parse_action


@registry.register_mas("autogen")
class AutoGenStyleMAS(BaseMAS):
    mas_type = "autogen"

    def __init__(self, mas_config: dict | None = None) -> None:
        self.mas_config = mas_config or {}
        self.assistant_agent = MASAgent(
            role="assistant agent",
            topology_node_id=0,
            system_prompt_template=AUTOGEN_ASSISTANT_SYSTEM_PROMPT_TEMPLATE,
            user_prompt_template=AUTOGEN_ASSISTANT_USER_PROMPT_TEMPLATE,
        )
        self.user_proxy_agent = MASAgent(
            role="user proxy agent",
            topology_node_id=1,
            system_prompt_template=AUTOGEN_USER_PROXY_SYSTEM_PROMPT_TEMPLATE,
            user_prompt_template=AUTOGEN_USER_PROXY_USER_PROMPT_TEMPLATE,
        )
        self.agents_list = [self.assistant_agent, self.user_proxy_agent]
        self.topology = {
            "assistant agent": [],
            "user proxy agent": ["assistant agent"],
        }

    def run(self, example: TaskExample, task, llm: BaseLLM, memory: MemoryProvider) -> Trajectory:
        graph = self._generate_graph(example, task, llm, memory, observation="")
        return self._trajectory_from_graph(
            example,
            llm,
            graph,
            {"mas_style": "autogen", "topology": self.topology, "edges": graph.edges},
        )

    def act(
        self,
        example: TaskExample,
        observation: str,
        task,
        llm: BaseLLM,
        memory: MemoryProvider,
        context: dict,
    ) -> ActionDecision:
        graph = self._generate_graph(example, task, llm, memory, observation=observation, context=context)
        graph.action = parse_action(str(graph.action or ""))
        return ActionDecision(
            action=str(graph.action),
            messages=graph.messages,
            metadata={"mas_style": "autogen", "topology": self.topology, "edges": graph.edges},
        )

    def _generate_graph(
        self,
        example: TaskExample,
        task,
        llm: BaseLLM,
        memory: MemoryProvider,
        observation: str,
        context: dict | None = None,
    ) -> MessageGraph:
        graph = MessageGraph(state=example.input)
        system_inputs = self._system_inputs(example)

        assistant_task_description, assistant_memory_metadata = self._task_description_with_memory(
            example,
            task,
            self.assistant_agent,
            {**(context or {}), "graph": graph.model_dump(mode="json"), "observation": observation},
            memory,
        )
        assistant_msg = self.assistant_agent.invoke(
            llm=llm,
            system_inputs=system_inputs,
            user_inputs={"task_description": assistant_task_description, "observation": observation},
            turn_id=0,
            parent_roles=None,
            metadata={"memory": assistant_memory_metadata},
            llm_kwargs=self._llm_kwargs(example, observation),
        )
        graph.update_message_graph(assistant_msg, self.assistant_agent.role, None)

        user_proxy_task_description, user_proxy_memory_metadata = self._task_description_with_memory(
            example,
            task,
            self.user_proxy_agent,
            {
                **(context or {}),
                "graph": graph.model_dump(mode="json"),
                "assistant_output": assistant_msg.content,
                "observation": observation,
            },
            memory,
        )
        user_proxy_msg = self.user_proxy_agent.invoke(
            llm=llm,
            system_inputs=system_inputs,
            user_inputs={"task_description": user_proxy_task_description, "assistant_output": assistant_msg.content},
            turn_id=1,
            parent_roles=[self.assistant_agent.role],
            metadata={"memory": user_proxy_memory_metadata},
            llm_kwargs=self._llm_kwargs(example, observation),
        )
        graph.update_message_graph(user_proxy_msg, self.user_proxy_agent.role, [self.assistant_agent.role])
        graph.action = user_proxy_msg.content
        return graph
