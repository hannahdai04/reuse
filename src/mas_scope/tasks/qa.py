"""QA task implementation."""

from __future__ import annotations

import json

from mas_scope.core.types import AgentSpec, TaskExample, Trajectory
from mas_scope.evaluation.qa_metrics import compute_qa_metrics
from mas_scope.tasks.base import BaseTask
from mas_scope.tools.answer_parser import extract_qa_answer


class QATask(BaseTask):
    task_type = "qa"

    def build_prompt(self, example: TaskExample, agent_spec: AgentSpec, context: dict) -> str:
        question = example.input.get("question")
        if not question:
            raise ValueError("QATask requires example.input['question'].")

        source_sections = []
        source_context = example.input.get("context")
        facts = example.input.get("facts")
        evidence = example.input.get("evidence")
        if source_context not in (None, "", [], {}):
            if isinstance(source_context, str):
                source_sections.append(f"context: {self._format_source(source_context)}")
            else:
                source_sections.append(f"context:\n{self._format_source(source_context)}")
        if facts not in (None, "", [], {}):
            source_sections.append(f"facts:\n{self._format_source(facts)}")
        if evidence not in (None, "", [], {}) and (facts in (None, "", [], {}) or example.dataset_name != "strategyqa"):
            source_sections.append(f"evidence:\n{self._format_source(evidence)}")

        source_text = "\n".join(source_sections) if source_sections else "None"
        dataset_instruction = self._dataset_instruction(example)
        memory_section = self._memory_section(context)
        return (
            f"Question: {question}\n"
            f"{memory_section}"
            f"Target Evidence:\n{source_text}\n"
            f"Task Requirements:\n{dataset_instruction}\n"
            "Output Contract: exactly one line: Final Answer: <answer>. "
            "No reasoning, citations, markdown, or extra text. Use a short span or yes/no when sufficient."
        )

    def parse_final_answer(self, trajectory: Trajectory) -> str | dict | None:
        if trajectory.final_answer is not None:
            return extract_qa_answer(trajectory.final_answer, trajectory.dataset_name)
        if trajectory.messages:
            return extract_qa_answer(trajectory.messages[-1].content, trajectory.dataset_name)
        return None

    def evaluate(self, prediction, example: TaskExample) -> dict:
        return compute_qa_metrics(prediction, example.target, example.dataset_name)

    def _format_source(self, value) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            if self._is_hotpot_context(value):
                sections = []
                for title, sentences in value:
                    sentence_text = " ".join(str(sentence).strip() for sentence in sentences if str(sentence).strip())
                    sections.append(f"- {title}: {sentence_text}")
                return "\n".join(sections)
            return "\n".join(f"- {self._format_source(item)}" for item in value)
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        return str(value)

    def _is_hotpot_context(self, value) -> bool:
        return all(
            isinstance(item, list)
            and len(item) == 2
            and isinstance(item[0], str)
            and isinstance(item[1], list)
            for item in value
        )

    def _dataset_instruction(self, example: TaskExample) -> str:
        dataset_name = example.dataset_name.lower()
        if dataset_name == "strategyqa":
            return "\n".join(
                [
                    "- This is a boolean QA task.",
                    "- The final answer must be exactly yes or no.",
                    "- Use the provided facts or evidence when available.",
                    "- Do not output explanations, caveats, citations, or full sentences.",
                    "- Answer yes only when the evidence supports the statement; otherwise answer no.",
                ]
            )
        if dataset_name == "hotpotqa":
            return "\n".join(
                [
                    "- This is a multi-hop evidence QA task.",
                    "- Use only Target Evidence as facts; memory is process guidance only.",
                    "- Connect the relevant evidence pieces before answering.",
                    "- Return the minimal supported answer span or entity.",
                    "- Do not stop at a bridge entity when the question asks for that entity's property, role, title, location, nationality, date, author, spouse, organization, or other attribute.",
                    "- For comparison or boolean questions, answer exactly yes or no when a boolean decision is requested.",
                    "- If multiple entities are mentioned, verify that the final answer matches the entity and attribute requested by the question.",
                    "- Before finalizing, check evidence support, answer type, answer granularity, and output format.",
                ]
            )
        return "\n".join(
            [
                "- Answer using the provided evidence.",
                "- Return the shortest correct final answer.",
            ]
        )
