"""In-house MacNet-style MAS with actor, critic, and summarizer agents."""

from __future__ import annotations

from mas_scope.core.registry import registry
from mas_scope.core.types import ActionDecision, MessageGraph, TaskExample, Trajectory
from mas_scope.llm.base import BaseLLM
from mas_scope.mas.agent import MASAgent
from mas_scope.mas.base import BaseMAS
from mas_scope.memory.base import MemoryProvider
from mas_scope.prompts.templates import (
    MACNET_ACTOR_SYSTEM_PROMPT_TEMPLATE,
    MACNET_ACTOR_USER_PROMPT_TEMPLATE,
    MACNET_CRITIC_SYSTEM_PROMPT_TEMPLATE,
    MACNET_CRITIC_USER_PROMPT_TEMPLATE,
    MACNET_FEEDBACK_PAGE_TEMPLATE,
    MACNET_SUMMARIZER_SYSTEM_PROMPT_TEMPLATE,
    MACNET_SUMMARIZER_USER_PROMPT_TEMPLATE,
)
from mas_scope.tools.action_parser import parse_action


@registry.register_mas("macnet")
class MacNetStyleMAS(BaseMAS):
    mas_type = "macnet"

    def __init__(self, mas_config: dict | None = None) -> None:
        self.mas_config = mas_config or {}
        self.actor_agent_1 = MASAgent("actor agent 1", 0, MACNET_ACTOR_SYSTEM_PROMPT_TEMPLATE, MACNET_ACTOR_USER_PROMPT_TEMPLATE)
        self.actor_agent_2 = MASAgent("actor agent 2", 0, MACNET_ACTOR_SYSTEM_PROMPT_TEMPLATE, MACNET_ACTOR_USER_PROMPT_TEMPLATE)
        self.critic_agent_1 = MASAgent("critic agent 1", 1, MACNET_CRITIC_SYSTEM_PROMPT_TEMPLATE, MACNET_CRITIC_USER_PROMPT_TEMPLATE)
        self.critic_agent_2 = MASAgent("critic agent 2", 1, MACNET_CRITIC_SYSTEM_PROMPT_TEMPLATE, MACNET_CRITIC_USER_PROMPT_TEMPLATE)
        self.summarizer_agent = MASAgent("summarizer agent", 2, MACNET_SUMMARIZER_SYSTEM_PROMPT_TEMPLATE, MACNET_SUMMARIZER_USER_PROMPT_TEMPLATE)
        self.agents_list = [
            self.actor_agent_1,
            self.actor_agent_2,
            self.critic_agent_1,
            self.critic_agent_2,
            self.summarizer_agent,
        ]
        self.agent_order = [
            "actor agent 1",
            "critic agent 1",
            "actor agent 2",
            "critic agent 2",
            "summarizer agent",
        ]
        self.topology = {
            "actor agent 1": [],
            "critic agent 1": ["actor agent 1"],
            "actor agent 2": [],
            "critic agent 2": ["actor agent 2"],
            "summarizer agent": ["critic agent 1", "critic agent 2"],
        }

    def run(self, example: TaskExample, task, llm: BaseLLM, memory: MemoryProvider) -> Trajectory:
        graph = self._generate_graph(example, task, llm, memory, observation="")
        return self._trajectory_from_graph(example, llm, graph, self._metadata(graph))

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
            metadata={
                **self._metadata(graph),
                "candidate_action_1": graph.metadata["candidate_action_1"],
                "candidate_action_2": graph.metadata["candidate_action_2"],
                "critic_feedback_1": graph.metadata["critic_feedback_1"],
                "critic_feedback_2": graph.metadata["critic_feedback_2"],
            },
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

        actor1_msg = self._invoke(graph, example, task, memory, self.actor_agent_1, llm, system_inputs, {
            "observation": observation,
        }, 0, None)
        critic1_msg = self._invoke(graph, example, task, memory, self.critic_agent_1, llm, system_inputs, {
            "actor_output": actor1_msg.content,
        }, 1, [self.actor_agent_1.role])
        actor2_msg = self._invoke(graph, example, task, memory, self.actor_agent_2, llm, system_inputs, {
            "observation": observation,
        }, 2, None)
        critic2_msg = self._invoke(graph, example, task, memory, self.critic_agent_2, llm, system_inputs, {
            "actor_output": actor2_msg.content,
        }, 3, [self.actor_agent_2.role])

        feedback_page1 = MACNET_FEEDBACK_PAGE_TEMPLATE.format(actor_output=actor1_msg.content, critic_output=critic1_msg.content)
        feedback_page2 = MACNET_FEEDBACK_PAGE_TEMPLATE.format(actor_output=actor2_msg.content, critic_output=critic2_msg.content)
        summarizer_msg = self._invoke(graph, example, task, memory, self.summarizer_agent, llm, system_inputs, {
            "feedback_page1": feedback_page1,
            "feedback_page2": feedback_page2,
        }, 4, [self.critic_agent_1.role, self.critic_agent_2.role])

        graph.action = summarizer_msg.content
        graph.metadata.update(
            {
                "candidate_action_1": actor1_msg.content,
                "candidate_action_2": actor2_msg.content,
                "critic_feedback_1": critic1_msg.content,
                "critic_feedback_2": critic2_msg.content,
            }
        )
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
            llm_kwargs=self._llm_kwargs(example, user_inputs.get("observation", "")),
        )
        graph.update_message_graph(msg, agent.role, parent_roles)
        return msg

    def _metadata(self, graph: MessageGraph) -> dict:
        return {
            "mas_style": "macnet",
            "topology": self.topology,
            "edges": graph.edges,
            "agent_order": self.agent_order,
        }
