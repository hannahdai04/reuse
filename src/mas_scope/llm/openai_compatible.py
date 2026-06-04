"""Optional OpenAI-compatible chat completion provider."""

from __future__ import annotations

import os
import time

import httpx

from mas_scope.core.env import load_environment
from mas_scope.core.exceptions import ConfigurationError
from mas_scope.llm.base import BaseLLM, LLMResponse


class OpenAICompatibleLLM(BaseLLM):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        timeout: float = 30.0,
        retries: int = 1,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        extra_body: dict | None = None,
    ) -> None:
        load_environment()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model_name = model_name or os.getenv("OPENAI_MODEL") or ""
        self.timeout = timeout
        self.retries = retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = extra_body or {}
        if not self.api_key:
            raise ConfigurationError("OPENAI_API_KEY is required for OpenAICompatibleLLM.")
        if not self.model_name:
            raise ConfigurationError("OPENAI_MODEL is required for OpenAICompatibleLLM.")

    def generate(self, messages: list[dict], **kwargs) -> LLMResponse:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
        }
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(kwargs.get("extra_body", self.extra_body))
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                raw = response.json()
                content = raw["choices"][0]["message"]["content"]
                return LLMResponse(content=content, usage=raw.get("usage", {}), raw=raw)
            except httpx.HTTPStatusError as exc:
                last_error = RuntimeError(self._format_http_error(exc))
            except Exception as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"OpenAI-compatible LLM request failed: {last_error}") from last_error

    def _format_http_error(self, exc: httpx.HTTPStatusError) -> str:
        response = exc.response
        body = response.text.strip()
        if len(body) > 800:
            body = body[:800] + "...[truncated]"
        return (
            f"HTTP {response.status_code} for {response.request.url} "
            f"body={body or '<empty>'}"
        )
