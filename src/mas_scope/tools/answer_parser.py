"""Lightweight output parsing for benchmark QA answers."""

from __future__ import annotations

import re

from mas_scope.tools.text_normalization import normalize_answer


ANSWER_PREFIX_RE = re.compile(
    r"^\s*(?:\*\*)?\s*(?:(?:final|answer|prediction|output[-\s]?answer)\s*)+[:：]\s*(?:\*\*)?\s*",
    re.IGNORECASE,
)
YES_NO_RE = re.compile(r"^\s*(?:\*\*)?\s*(yes|no)\b", re.IGNORECASE)
NOANSWER_RE = re.compile(r"^\s*(?:\*\*)?\s*(noanswer|no\s+answer)\b", re.IGNORECASE)


def strip_answer_prefix(text: str) -> str:
    value = text.strip()
    while True:
        updated = ANSWER_PREFIX_RE.sub("", value).strip()
        if updated == value:
            return updated
        value = updated


def extract_binary_answer(text: str) -> str | None:
    value = strip_answer_prefix(text)
    match = NOANSWER_RE.match(value)
    if match:
        return "noanswer"
    match = YES_NO_RE.match(value)
    if match:
        return match.group(1).lower()
    return None


def extract_qa_answer(text, dataset_name: str | None = None) -> str:
    """Extract a task answer from common LLM final-answer formats.

    This intentionally avoids semantic matching against the gold answer. The LLM
    still has to solve the task; the parser only removes protocol noise.
    """

    value = "" if text is None else str(text).strip()
    if not value:
        return ""
    value = _first_content_line(value)
    value = strip_answer_prefix(value)
    if (dataset_name or "").lower() == "strategyqa":
        binary = extract_binary_answer(value)
        if binary in {"yes", "no"}:
            return binary
    return value.strip()


def prediction_for_scoring(prediction, target: dict, dataset_name: str | None = None) -> str:
    value = extract_qa_answer(prediction, dataset_name)
    answer = target.get("normalized_answer") or target.get("answer")
    normalized_gold = normalize_answer(answer)
    if normalized_gold in {"yes", "no", "noanswer"}:
        binary = extract_binary_answer(value)
        if binary is not None:
            return binary
    return value


def _first_content_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else text.strip()
