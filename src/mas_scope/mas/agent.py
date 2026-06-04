"""Agent schemas and the lightweight in-house MAS agent."""

from __future__ import annotations

from uuid import uuid4

from mas_scope.core.types import AgentMessage, AgentSpec
from mas_scope.llm.base import BaseLLM


class MASAgent:
    """Small prompt-formatting agent inspired by LLMAgent-style APIs."""

    def __init__(
        self,
        role: str,
        topology_node_id: int,
        system_prompt_template: str,
        user_prompt_template: str,
        name: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        self.role = role
        self.name = name or role
        self.id = agent_id or f"{role.replace(' ', '_')}_{uuid4().hex[:8]}"
        self.topology_node_id = topology_node_id
        self.system_prompt_template = system_prompt_template
        self.user_prompt_template = user_prompt_template

    def invoke(
        self,
        llm: BaseLLM,
        system_inputs: dict,
        user_inputs: dict,
        turn_id: int,
        parent_roles: list[str] | None = None,
        metadata: dict | None = None,
        llm_kwargs: dict | None = None,
    ) -> AgentMessage:
        system_prompt = self._format_template("system_prompt_template", self.system_prompt_template, system_inputs)
        user_prompt = self._format_template("user_prompt_template", self.user_prompt_template, user_inputs)
        response = llm.generate(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **(llm_kwargs or {}),
        )
        return AgentMessage(
            turn_id=turn_id,
            agent_name=self.name,
            role=self.role,
            content=response.content,
            metadata={
                "agent_id": self.id,
                "topology_node_id": self.topology_node_id,
                "parent_roles": parent_roles or [],
                "system_input_keys": sorted(system_inputs.keys()),
                "user_input_keys": sorted(user_inputs.keys()),
                "llm_usage": response.usage,
                "llm_raw": response.raw,
                **(metadata or {}),
            },
        )

    def _format_template(self, template_name: str, template: str, values: dict) -> str:
        try:
            return template.format(**values)
        except KeyError as exc:
            missing = exc.args[0]
            raise ValueError(
                f"MASAgent '{self.role}' {template_name} missing field '{missing}'."
            ) from exc


__all__ = ["AgentSpec", "MASAgent"]
