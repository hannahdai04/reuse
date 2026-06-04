"""JSON parsing helpers for allocator LLM outputs."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json_object(content: str) -> dict[str, Any]:
    payload = _parse_json(content)
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object.")
    return payload


def parse_json_array(content: str) -> list[Any]:
    payload = _parse_json(content)
    if not isinstance(payload, list):
        raise ValueError("Expected a JSON array.")
    return payload


def _parse_json(content: str) -> Any:
    text = content.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1).strip())

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
    raise ValueError("No JSON payload found.")
