"""Deterministic offline LLM for tests and smoke runs."""

from __future__ import annotations

from mas_scope.llm.base import BaseLLM, LLMResponse


class MockLLM(BaseLLM):
    def __init__(self, model_name: str = "mock-llm") -> None:
        self.model_name = model_name

    def generate(self, messages: list[dict], **kwargs) -> LLMResponse:
        task_type = kwargs.get("task_type")
        observation = str(kwargs.get("observation") or "").lower()
        target = kwargs.get("target") or {}
        prompt_text = "\n".join(str(message.get("content", "")) for message in messages).lower()

        if task_type == "interactive":
            content = self._interactive_action(observation or prompt_text)
        elif task_type == "formal_planning":
            content = self._planning_answer(target)
        elif task_type == "qa":
            content = self._qa_answer(target)
        elif "parenthesized action" in prompt_text or "pddl" in prompt_text:
            content = self._planning_answer(target)
        elif "final action" in prompt_text or "candidate action" in prompt_text or "next action" in prompt_text:
            content = self._interactive_action(observation or prompt_text)
        elif "answer" in prompt_text or "question:" in prompt_text:
            content = self._qa_answer(target)
        else:
            content = "unknown"

        return LLMResponse(
            content=content,
            usage={"prompt_tokens": len(prompt_text.split()), "completion_tokens": len(content.split())},
            raw={"provider": "mock"},
        )

    def _qa_answer(self, target: dict) -> str:
        if "normalized_answer" in target and target["normalized_answer"]:
            return str(target["normalized_answer"])
        if "answer" in target and target["answer"] is not None:
            return str(target["answer"])
        return "unknown"

    def _planning_answer(self, target: dict) -> str:
        reference_plan = target.get("reference_plan")
        if isinstance(reference_plan, list) and reference_plan:
            return "\n".join(str(action) for action in reference_plan)
        if isinstance(reference_plan, str) and reference_plan.strip():
            return reference_plan
        return "(move robot room-a room-b)"

    def _interactive_action(self, text: str) -> str:
        if "kitchen" in text or "apple" in text:
            if "holding the apple" in text or "take the apple" in text:
                return "put apple in fridge"
            return "take apple"
        if "lab" in text or "thermometer" in text or "water" in text:
            if "take the thermometer" in text or "thermometer" in text and "cup of water" not in text:
                return "measure water temperature"
            return "take thermometer"
        return "look"
