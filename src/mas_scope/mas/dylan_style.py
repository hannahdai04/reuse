"""In-house DyLAN-style dynamically selected MAS."""

from __future__ import annotations

from mas_scope.core.registry import registry
from mas_scope.core.types import ActionDecision, MessageGraph, TaskExample, Trajectory
from mas_scope.llm.base import BaseLLM
from mas_scope.mas.agent import MASAgent
from mas_scope.mas.base import BaseMAS
from mas_scope.memory.base import MemoryProvider
from mas_scope.prompts.templates import (
    DYLAN_FINALIZER_SYSTEM_PROMPT_TEMPLATE,
    DYLAN_FINALIZER_USER_PROMPT_TEMPLATE,
    DYLAN_PLANNER_SYSTEM_PROMPT_TEMPLATE,
    DYLAN_PLANNER_USER_PROMPT_TEMPLATE,
    DYLAN_REASONER_SYSTEM_PROMPT_TEMPLATE,
    DYLAN_REASONER_USER_PROMPT_TEMPLATE,
    DYLAN_VERIFIER_SYSTEM_PROMPT_TEMPLATE,
    DYLAN_VERIFIER_USER_PROMPT_TEMPLATE,
)
from mas_scope.tools.action_parser import parse_action


@registry.register_mas("dylan")
class DyLANStyleMAS(BaseMAS):
    mas_type = "dylan"

    def __init__(self, mas_config: dict | None = None) -> None:
        self.mas_config = mas_config or {}
        self.planner_agent = MASAgent("planner agent", 0, DYLAN_PLANNER_SYSTEM_PROMPT_TEMPLATE, DYLAN_PLANNER_USER_PROMPT_TEMPLATE)
        self.reasoner_agent = MASAgent("reasoner agent", 1, DYLAN_REASONER_SYSTEM_PROMPT_TEMPLATE, DYLAN_REASONER_USER_PROMPT_TEMPLATE)
        self.verifier_agent = MASAgent("verifier agent", 2, DYLAN_VERIFIER_SYSTEM_PROMPT_TEMPLATE, DYLAN_VERIFIER_USER_PROMPT_TEMPLATE)
        self.finalizer_agent = MASAgent("finalizer agent", 3, DYLAN_FINALIZER_SYSTEM_PROMPT_TEMPLATE, DYLAN_FINALIZER_USER_PROMPT_TEMPLATE)
        self.agents_list = [self.planner_agent, self.reasoner_agent, self.verifier_agent, self.finalizer_agent]
        self.topology = {
            "planner agent": [],
            "reasoner agent": ["planner agent"],
            "verifier agent": ["reasoner agent"],
            "finalizer agent": ["reasoner agent", "verifier agent"],
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
        selected_agents: list[str] = []
        selection_trace: list[dict] = []

        planner_msg = self._invoke(graph, selected_agents, selection_trace, example, task, memory, self.planner_agent, llm, system_inputs, {
            "observation": observation,
        }, 0, None, "initial planning")
        reasoner_msg = self._invoke(graph, selected_agents, selection_trace, example, task, memory, self.reasoner_agent, llm, system_inputs, {
            "plan": planner_msg.content,
            "verifier_feedback": "",
        }, 1, [self.planner_agent.role], "reason from plan")
        verifier_msg = self._invoke(graph, selected_agents, selection_trace, example, task, memory, self.verifier_agent, llm, system_inputs, {
            "candidate_output": reasoner_msg.content,
        }, 2, [self.reasoner_agent.role], "verify candidate")

        verifier_text = verifier_msg.content.lower()
        if any(token in verifier_text for token in ("invalid", "incorrect", "revise")):
            reasoner_msg = self._invoke(graph, selected_agents, selection_trace, example, task, memory, self.reasoner_agent, llm, system_inputs, {
                "plan": planner_msg.content,
                "verifier_feedback": verifier_msg.content,
            }, 3, [self.verifier_agent.role], "revise after verifier")

        final_msg = self._invoke(graph, selected_agents, selection_trace, example, task, memory, self.finalizer_agent, llm, system_inputs, {
            "answer": reasoner_msg.content,
            "verifier_output": verifier_msg.content,
        }, len(graph.messages), [self.reasoner_agent.role, self.verifier_agent.role], "finalize")
        graph.action = final_msg.content
        graph.metadata["selected_agents"] = selected_agents
        graph.metadata["selection_trace"] = selection_trace
        return graph

    def _invoke(
        self,
        graph,
        selected_agents,
        selection_trace,
        example,
        task,
        memory,
        agent,
        llm,
        system_inputs,
        user_inputs,
        turn_id,
        parent_roles,
        reason,
    ):
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
            metadata={"selection_reason": reason, "memory": memory_metadata},
            llm_kwargs=self._llm_kwargs(example, user_inputs.get("observation", "")),
        )
        graph.update_message_graph(msg, agent.role, parent_roles)
        selected_agents.append(agent.role)
        selection_trace.append({"turn_id": turn_id, "agent": agent.role, "reason": reason})
        return msg

    def _metadata(self, graph: MessageGraph) -> dict:
        return {
            "mas_style": "dylan",
            "topology": self.topology,
            "edges": graph.edges,
            "selected_agents": graph.metadata.get("selected_agents", []),
            "selection_trace": graph.metadata.get("selection_trace", []),
        }
