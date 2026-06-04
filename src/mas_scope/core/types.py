"""Central shared schemas for the mas_scope package."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TaskExample(BaseModel):
    example_id: str
    dataset_name: str
    task_type: Literal["qa", "interactive", "formal_planning"]
    split: str
    input: dict
    target: dict
    metadata: dict = Field(default_factory=dict)


class AgentSpec(BaseModel):
    name: str
    role: str
    system_prompt: str
    tools: list[str] = Field(default_factory=list)
    max_turns: int = 3


class AgentMessage(BaseModel):
    turn_id: int
    agent_name: str
    role: str
    content: str
    metadata: dict = Field(default_factory=dict)


class ActionDecision(BaseModel):
    action: str
    messages: list[AgentMessage]
    metadata: dict = Field(default_factory=dict)


class EnvironmentState(BaseModel):
    episode_id: str
    step_id: int
    observation: str
    admissible_actions: list[str] | None = None
    score: float | None = None
    done: bool = False
    metadata: dict = Field(default_factory=dict)


class EnvironmentStep(BaseModel):
    episode_id: str
    step_id: int
    action: str
    observation: str
    reward: float | None = None
    done: bool = False
    info: dict = Field(default_factory=dict)


class Trajectory(BaseModel):
    run_id: str
    example_id: str
    dataset_name: str
    mas_type: str
    model_name: str
    messages: list[AgentMessage]
    environment_steps: list[EnvironmentStep] = Field(default_factory=list)
    final_answer: str | dict | None = None
    error: str | None = None
    usage: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)


class TaskResult(BaseModel):
    run_id: str
    example_id: str
    prediction: str | dict | None
    target: dict
    metrics: dict
    success: bool
    cost: dict = Field(default_factory=dict)
    error: str | None = None


class MessageGraph(BaseModel):
    state: str | dict
    messages: list[AgentMessage] = Field(default_factory=list)
    edges: list[dict] = Field(default_factory=list)
    action: str | dict | None = None
    metadata: dict = Field(default_factory=dict)

    def update_message_graph(
        self,
        message: AgentMessage,
        role: str,
        parent_roles: list[str] | None,
    ) -> None:
        self.messages.append(message)
        for parent_role in parent_roles or []:
            self.edges.append({"source": parent_role, "target": role})
