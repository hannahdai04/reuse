import pytest

from mas_scope.core.types import AgentMessage, AgentSpec, TaskExample, Trajectory
from mas_scope.evaluation.qa_metrics import compute_qa_metrics
from mas_scope.tasks.qa import QATask
from mas_scope.tools.answer_parser import extract_qa_answer


def example_with_input(input_payload):
    return TaskExample(
        example_id="ex1",
        dataset_name="hotpotqa",
        task_type="qa",
        split="dev",
        input=input_payload,
        target={"answer": "Paris"},
    )


def agent():
    return AgentSpec(name="Reasoner", role="reasoner", system_prompt="Answer carefully.")


def test_build_prompt_formats_context_string():
    prompt = QATask().build_prompt(example_with_input({"question": "Capital?", "context": "France context."}), agent(), {})

    assert "Question: Capital?" in prompt
    assert "context: France context." in prompt
    assert "Final Answer: <answer>" in prompt


def test_build_prompt_formats_context_list():
    prompt = QATask().build_prompt(example_with_input({"question": "Capital?", "context": ["France", {"city": "Paris"}]}), agent(), {})

    assert "- France" in prompt
    assert '{"city": "Paris"}' in prompt


def test_build_prompt_formats_facts_and_evidence_list_or_dict():
    prompt = QATask().build_prompt(
        example_with_input(
            {
                "question": "Is it true?",
                "facts": ["fact one", "fact two"],
                "evidence": {"source": "sample", "score": 1},
            }
        ),
        agent(),
        {},
    )

    assert "facts:" in prompt
    assert "- fact one" in prompt
    assert "evidence:" in prompt
    assert '"source": "sample"' in prompt


def test_build_prompt_missing_question_raises_value_error():
    with pytest.raises(ValueError, match="question"):
        QATask().build_prompt(example_with_input({"context": "no question"}), agent(), {})


def test_strategyqa_prompt_requires_yes_no():
    example = TaskExample(
        example_id="s1",
        dataset_name="strategyqa",
        task_type="qa",
        split="dev",
        input={"question": "Can penguins fly?"},
        target={"normalized_answer": "no"},
    )

    prompt = QATask().build_prompt(example, agent(), {})

    assert "answer must be exactly yes or no" in prompt


def test_parse_final_answer_fallback_strips_whitespace():
    trajectory = Trajectory(
        run_id="run1",
        example_id="ex1",
        dataset_name="hotpotqa",
        mas_type="autogen",
        model_name="mock",
        messages=[AgentMessage(turn_id=0, agent_name="Finalizer", role="finalizer", content="  Paris \n")],
    )

    assert QATask().parse_final_answer(trajectory) == "Paris"


def test_parse_final_answer_strips_answer_prefix():
    trajectory = Trajectory(
        run_id="run1",
        example_id="ex1",
        dataset_name="hotpotqa",
        mas_type="autogen",
        model_name="mock",
        messages=[],
        final_answer="**Answer:** Chief of Protocol",
    )

    assert QATask().parse_final_answer(trajectory) == "Chief of Protocol"


def test_strategyqa_extracts_leading_yes_no():
    assert extract_qa_answer("No, Albany, GA will not reach it first.", "strategyqa") == "no"
    assert extract_qa_answer("**Answer:** Yes, it is English-based.", "strategyqa") == "yes"


def test_strategyqa_metrics_score_long_yes_no_outputs():
    metrics = compute_qa_metrics(
        "No, Albany, GA will not reach it first.",
        {"answer": False, "normalized_answer": "no"},
        dataset_name="strategyqa",
    )

    assert metrics["exact_match"] == 1.0
    assert metrics["yes_no_accuracy"] == 1.0


def test_hotpotqa_yes_no_scoring_uses_binary_extraction():
    metrics = compute_qa_metrics(
        "Yes, both people are American.",
        {"answer": "yes"},
        dataset_name="hotpotqa",
    )

    assert metrics["exact_match"] == 1.0
    assert metrics["token_f1"] == 1.0
