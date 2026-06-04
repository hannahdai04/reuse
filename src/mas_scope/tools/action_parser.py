"""Action parser for LLM outputs."""

from __future__ import annotations


def parse_action(text: str) -> str:
    raw = text.strip()
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return raw

    content = lines[0]
    prefixes = ("final action:", "action:", "final answer:", "answer:")
    lowered = content.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            content = content[len(prefix) :].strip()
            break
    return content or raw
