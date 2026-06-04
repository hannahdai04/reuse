"""Validation and normalization helpers for dataset adapters."""

from __future__ import annotations


def require_fields(example: dict, fields: list[str]) -> list[str]:
    errors: list[str] = []
    for field in fields:
        if field not in example or example[field] in (None, ""):
            errors.append(f"missing required field: {field}")
    return errors


def normalize_yes_no(value) -> str | None:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)) and value in (0, 1):
        return "yes" if int(value) == 1 else "no"
    if isinstance(value, str):
        text = value.strip().lower()
        mapping = {
            "yes": "yes",
            "y": "yes",
            "true": "yes",
            "1": "yes",
            "no": "no",
            "n": "no",
            "false": "no",
            "0": "no",
        }
        return mapping.get(text)
    return None
