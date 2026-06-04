"""LLM base interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    content: str
    usage: dict = Field(default_factory=dict)
    raw: dict = Field(default_factory=dict)


class BaseLLM(ABC):
    model_name: str

    @abstractmethod
    def generate(self, messages: list[dict], **kwargs) -> LLMResponse:
        ...
