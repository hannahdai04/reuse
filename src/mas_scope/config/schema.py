"""Experiment config schema."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    builder: str
    data_path: Path
    split: str = "dev"
    limit: int | None = None
    offset: int = 0


class TaskConfig(BaseModel):
    type: Literal["qa", "interactive", "formal_planning"]
    max_steps: int = 50


class MASConfig(BaseModel):
    type: Literal["autogen", "macnet", "camel", "dylan"]
    config: dict = Field(default_factory=dict)


class LLMConfig(BaseModel):
    provider: Literal["mock", "openai-compatible"] = "mock"
    model: str = "mock-llm"
    timeout: float = 30.0
    retries: int = 1
    temperature: float = 0.0
    max_tokens: int | None = None
    extra_body: dict = Field(default_factory=dict)


class MemoryConfig(BaseModel):
    provider: Literal["null", "llm-scope"] | None = None
    config: dict = Field(default_factory=dict)


class EnvironmentConfig(BaseModel):
    provider: str | None = None
    config: dict = Field(default_factory=dict)


class OutputConfig(BaseModel):
    dir: Path = Path("runs")


class ExperimentConfig(BaseModel):
    experiment_name: str
    dataset: DatasetConfig
    task: TaskConfig
    mas: MASConfig
    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
