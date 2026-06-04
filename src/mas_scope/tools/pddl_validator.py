"""Lightweight PDDL plan format checks and prompt helpers."""

from __future__ import annotations

import re


ACTION_RE = re.compile(r"^\([A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+)*\)$")
ACTION_FIND_RE = re.compile(r"\([A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+)*\)")
ACTION_SCHEMA_RE = re.compile(r"\(:action\s+([A-Za-z0-9_-]+)", re.IGNORECASE)
VARIABLE_RE = re.compile(r"\?[A-Za-z0-9_-]+")
TOKEN_RE = re.compile(r"-|[A-Za-z0-9_?][A-Za-z0-9_?-]*")
ANSWER_PREFIX_RE = re.compile(r"^(?:final\s+answer|answer|final\s+action|action)\s*:\s*", re.IGNORECASE)
BULLET_PREFIX_RE = re.compile(r"^(?:[-*]\s+|\d+[\).\s]+)")


def split_plan(plan: str | list[str] | None) -> list[str]:
    if plan is None:
        return []
    if isinstance(plan, list):
        return [str(item).strip() for item in plan if str(item).strip()]
    return [line.strip() for line in str(plan).splitlines() if line.strip()]


def clean_plan_output(plan: str | list[str] | None) -> str | list[str] | None:
    """Extract complete grounded action lines from a model response.

    The raw response remains available in the trajectory. This function only
    normalizes the prediction used by lightweight Phase-1 metrics.
    """
    if plan is None or isinstance(plan, list):
        return plan

    text = str(plan).strip()
    actions: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("```") or line.upper() in {"[PLAN]", "[PLAN END]"}:
            continue
        had_answer_prefix = ANSWER_PREFIX_RE.match(line) is not None
        line = ANSWER_PREFIX_RE.sub("", line).strip()
        line = BULLET_PREFIX_RE.sub("", line).strip()
        for action in ACTION_FIND_RE.findall(line):
            if had_answer_prefix and actions and action == actions[-1]:
                continue
            actions.append(action)
    return "\n".join(actions) if actions else text


def valid_action_format(plan: str | list[str] | None) -> bool:
    actions = split_plan(plan)
    return bool(actions) and all(ACTION_RE.match(action) for action in actions)


def extract_action_names(domain_pddl: str | None) -> list[str]:
    if not domain_pddl:
        return []
    return sorted(set(ACTION_SCHEMA_RE.findall(domain_pddl)))


def extract_action_signatures(domain_pddl: str | None) -> list[dict]:
    """Return lightweight action signatures from a PDDL domain.

    This is intentionally not a full PDDL parser. It extracts enough structure
    for prompt constraints and offline smoke metrics without adding a planner
    dependency.
    """
    if not domain_pddl:
        return []

    clean = strip_pddl_comments(domain_pddl)
    matches = list(ACTION_SCHEMA_RE.finditer(clean))
    signatures = []
    for index, match in enumerate(matches):
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(clean)
        block = clean[match.end() : next_start]
        parameters = _extract_parameters(block)
        signatures.append({"name": match.group(1), "parameters": parameters, "arity": len(parameters)})
    return signatures


def format_action_signatures(domain_pddl: str | None) -> str:
    signatures = extract_action_signatures(domain_pddl)
    if not signatures:
        names = extract_action_names(domain_pddl)
        return ", ".join(names) if names else "unknown"
    return ", ".join(f"{item['name']}/{item['arity']}" for item in signatures)


def extract_declared_objects(problem_pddl: str | None, domain_pddl: str | None = None) -> list[str]:
    """Extract problem objects plus domain constants, ignoring PDDL type names."""
    objects = _extract_typed_names(_section_body(problem_pddl, "objects"))
    constants = _extract_typed_names(_section_body(domain_pddl, "constants"))
    return sorted(set(objects + constants))


def format_declared_objects(problem_pddl: str | None, domain_pddl: str | None = None, limit: int = 80) -> str:
    objects = extract_declared_objects(problem_pddl, domain_pddl)
    if not objects:
        return "unknown"
    if len(objects) <= limit:
        return ", ".join(objects)
    shown = ", ".join(objects[:limit])
    return f"{shown}, ... ({len(objects)} total)"


def no_variables_in_plan(plan: str | list[str] | None) -> bool:
    actions = split_plan(plan)
    return bool(actions) and not any(VARIABLE_RE.search(action) for action in actions)


def plan_uses_known_actions(plan: str | list[str] | None, action_names: list[str] | None) -> bool:
    if not action_names:
        return True
    allowed = set(action_names)
    actions = split_plan(plan)
    parsed = [_parse_action_line(action) for action in actions]
    return bool(parsed) and all(item is not None and item[0] in allowed for item in parsed)


def plan_uses_declared_objects(plan: str | list[str] | None, objects: list[str] | None) -> bool:
    if not objects:
        return True
    allowed = set(objects)
    actions = split_plan(plan)
    parsed = [_parse_action_line(action) for action in actions]
    return bool(parsed) and all(item is not None and all(arg in allowed for arg in item[1]) for item in parsed)


def strip_pddl_comments(pddl: str | None) -> str:
    if not pddl:
        return ""
    return "\n".join(line.split(";", 1)[0] for line in pddl.splitlines())


def _extract_parameters(action_block: str) -> list[str]:
    match = re.search(r":parameters\s*\(", action_block, re.IGNORECASE)
    if not match:
        return []
    start = match.end() - 1
    end = _find_balanced_end(action_block, start)
    if end is None:
        return []
    body = action_block[start + 1 : end - 1]
    return _extract_typed_names(body)


def _section_body(pddl: str | None, section_name: str) -> str:
    clean = strip_pddl_comments(pddl)
    match = re.search(rf"\(:{re.escape(section_name)}\b", clean, re.IGNORECASE)
    if not match:
        return ""
    end = _find_balanced_end(clean, match.start())
    if end is None:
        return ""
    return clean[match.end() : end - 1]


def _find_balanced_end(text: str, start: int) -> int | None:
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def _extract_typed_names(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(text)
    names: list[str] = []
    skip_type = False
    for token in tokens:
        if token == "-":
            skip_type = True
            continue
        if skip_type:
            skip_type = False
            continue
        names.append(token)
    return names


def _parse_action_line(action: str) -> tuple[str, list[str]] | None:
    if not ACTION_RE.match(action):
        return None
    tokens = action.strip()[1:-1].split()
    if not tokens:
        return None
    return tokens[0], tokens[1:]
