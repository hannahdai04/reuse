import json
from pathlib import Path

from mas_scope.llm.base import LLMResponse

from extract_memories import (
    TrajectoryInput,
    build_messages,
    load_trajectories,
    parse_json_array,
    run_extraction,
)


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[list[dict]] = []

    def generate(self, messages: list[dict], **kwargs) -> LLMResponse:
        self.calls.append(messages)
        if not self.responses:
            raise RuntimeError("No fake response available.")
        return LLMResponse(content=self.responses.pop(0))


class RaisingLLM:
    def generate(self, messages: list[dict], **kwargs) -> LLMResponse:
        raise RuntimeError("network failed")


def _memory_response(**overrides):
    item = {
        "source": {
            "producer_agent": "planner",
            "producer_role": "planner",
        },
        "source_task_description": "A source task where agents decomposed a question.",
        "condition": "When a future task requires decomposing a broad objective into checkable subtasks.",
        "experience": "Split the objective into independent subtasks before asking critics to verify the result.",
        "evidence": "The trajectory succeeded after the planner separated the work and the critic checked each part.",
    }
    item.update(overrides)
    return json.dumps([item])


def test_load_trajectories_supports_json_jsonl_and_txt(tmp_path: Path):
    input_dir = tmp_path / "trajectories"
    input_dir.mkdir()
    (input_dir / "one.json").write_text(json.dumps({"run_id": "r1", "messages": []}), encoding="utf-8")
    (input_dir / "many.jsonl").write_text(
        json.dumps({"example_id": "e1", "messages": []}) + "\nnot-json\n",
        encoding="utf-8",
    )
    (input_dir / "raw.txt").write_text("plain trajectory text", encoding="utf-8")

    trajectories = load_trajectories(input_dir)

    assert [trajectory.trajectory_id for trajectory in trajectories] == [
        "e1",
        "many_000002",
        "r1",
        "raw",
    ]
    assert trajectories[1].content == "not-json"
    assert trajectories[1].source_line == 2


def test_load_trajectories_enriches_mas_run_artifacts(tmp_path: Path):
    input_dir = tmp_path / "run"
    input_dir.mkdir()
    (input_dir / "trajectories.jsonl").write_text(
        json.dumps({"run_id": "run_1", "example_id": "ex_1", "messages": []}) + "\n",
        encoding="utf-8",
    )
    (input_dir / "examples.jsonl").write_text(
        json.dumps({"example_id": "ex_1", "question": "What happened?"}) + "\n",
        encoding="utf-8",
    )
    (input_dir / "results.jsonl").write_text(
        json.dumps({"run_id": "run_1", "success": True, "final_answer": "yes"}) + "\n",
        encoding="utf-8",
    )
    (input_dir / "config.yaml").write_text("ignored: true", encoding="utf-8")

    trajectories = load_trajectories(input_dir)

    assert len(trajectories) == 1
    payload = json.loads(trajectories[0].content)
    assert payload["trajectory_id"] == "run_1"
    assert payload["task"]["input"] == {"question": "What happened?"}
    assert payload["result"]["success"] is True
    assert "llm_raw" not in payload


def test_load_trajectories_compacts_hotpotqa_context_and_team_metadata(tmp_path: Path):
    input_dir = tmp_path / "run"
    input_dir.mkdir()
    (input_dir / "trajectories.jsonl").write_text(
        json.dumps(
            {
                "run_id": "run_hotpot",
                "example_id": "hotpot_1",
                "dataset_name": "hotpotqa",
                "mas_type": "macnet",
                "messages": [
                    {
                        "turn_id": 0,
                        "agent_name": "actor agent 1",
                        "role": "actor agent 1",
                        "content": "Final Answer: United States ambassador to Ghana",
                        "metadata": {"llm_raw": {"provider": "mock"}},
                    }
                ],
                "metadata": {
                    "mas_style": "macnet",
                    "agent_order": ["actor agent 1", "critic agent 1", "summarizer agent"],
                    "topology": {"critic agent 1": ["actor agent 1"]},
                    "edges": [{"source": "actor agent 1", "target": "critic agent 1"}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (input_dir / "examples.jsonl").write_text(
        json.dumps(
            {
                "example_id": "hotpot_1",
                "dataset_name": "hotpotqa",
                "task_type": "qa",
                "input": {
                    "question": "What government position was held by the woman who portrayed Corliss Archer?",
                    "context": [
                        ["Kiss and Tell", ["Kiss and Tell starred Shirley Temple as Corliss Archer."]],
                        [
                            "Shirley Temple",
                            [
                                "Shirley Temple was an actress and diplomat.",
                                "She served as Chief of Protocol of the United States.",
                            ],
                        ],
                    ],
                },
                "target": {
                    "answer": "Chief of Protocol",
                    "supporting_facts": [["Kiss and Tell", 0], ["Shirley Temple", 1]],
                },
                "metadata": {"question_type": "bridge", "level": "hard", "benchmark_format": "hotpotqa"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (input_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "run_id": "run_hotpot",
                "example_id": "hotpot_1",
                "prediction": "United States ambassador to Ghana",
                "target": {"answer": "Chief of Protocol"},
                "metrics": {"exact_match": 0.0},
                "success": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    trajectories = load_trajectories(input_dir)

    payload = json.loads(trajectories[0].content)
    context_summary = payload["task"]["input"]["context_summary"]
    assert payload["dataset_name"] == "hotpotqa"
    assert payload["team_metadata"]["agent_order"] == ["actor agent 1", "critic agent 1", "summarizer agent"]
    assert payload["team_metadata"]["edges"] == [{"source": "actor agent 1", "target": "critic agent 1"}]
    assert payload["task"]["metadata"]["question_type"] == "bridge"
    assert context_summary["num_context_articles"] == 2
    assert context_summary["supporting_sentences"][1]["sentence"] == "She served as Chief of Protocol of the United States."
    assert payload["messages"][0]["role"] == "actor agent 1"
    assert "metadata" not in payload["messages"][0]
    assert trajectories[0].metadata["success"] is False


def test_parse_json_array_accepts_fenced_and_extra_text():
    assert parse_json_array('[{"a": 1}]') == [{"a": 1}]
    assert parse_json_array('```json\n[{"a": 2}]\n```') == [{"a": 2}]
    assert parse_json_array('Here is the array:\n[{"a": 3}]\nDone') == [{"a": 3}]


def test_build_messages_uses_hotpotqa_failure_and_multi_agent_instructions():
    payload = {
        "trajectory_id": "traj_fail",
        "example_id": "hotpot_1",
        "dataset_name": "hotpotqa",
        "mas_type": "macnet",
        "task": {
            "example_id": "hotpot_1",
            "dataset_name": "hotpotqa",
            "task_type": "qa",
            "metadata": {"question_type": "bridge"},
        },
        "messages": [
            {"turn_id": 0, "agent_name": "actor agent 1", "role": "actor agent 1", "content": "wrong"},
            {"turn_id": 1, "agent_name": "critic agent 1", "role": "critic agent 1", "content": "accept"},
        ],
        "team_metadata": {
            "topology": {"critic agent 1": ["actor agent 1"]},
            "agent_order": ["actor agent 1", "critic agent 1"],
        },
        "result": {
            "prediction": "wrong",
            "target": {"answer": "right"},
            "metrics": {"exact_match": 0.0},
            "success": False,
        },
    }

    messages = build_messages(TrajectoryInput("traj_fail", "run/trajectories.jsonl", json.dumps(payload)), max_chars=60000)

    system_prompt = messages[0]["content"]
    user_prompt = messages[1]["content"]
    assert "Failed trajectory focus" in system_prompt
    assert "HotpotQA-specific extraction focus" in system_prompt
    assert "Multi-agent extraction focus" in system_prompt
    assert '"outcome": "failure"' in user_prompt
    assert '"hotpotqa": true' in user_prompt
    assert '"agent_order": [' in user_prompt


def test_build_messages_uses_success_instruction_for_successful_runs():
    payload = {
        "trajectory_id": "traj_success",
        "dataset_name": "hotpotqa",
        "result": {"metrics": {"exact_match": 1.0}, "success": True},
    }

    messages = build_messages(TrajectoryInput("traj_success", "run/trajectories.jsonl", json.dumps(payload)), max_chars=60000)

    assert "Successful trajectory focus" in messages[0]["content"]
    assert '"outcome": "success"' in messages[1]["content"]


def test_run_extraction_writes_whitelisted_memories_and_sequential_ids(tmp_path: Path):
    input_dir = tmp_path / "trajectories"
    input_dir.mkdir()
    (input_dir / "a.jsonl").write_text(
        json.dumps({"trajectory_id": "traj_a", "messages": []})
        + "\n"
        + json.dumps({"trajectory_id": "traj_b", "messages": []})
        + "\n",
        encoding="utf-8",
    )
    responses = [
        _memory_response(scope="assistant-only", agent_mask=[1, 0], target_agent="assistant"),
        _memory_response(source={"producer_agent": "multi-agent", "producer_role": "team"}),
    ]
    llm = FakeLLM(responses)
    output_file = tmp_path / "memories.jsonl"

    stats = run_extraction(
        input_dir=input_dir,
        output_file=output_file,
        llm=llm,
        logs_dir=tmp_path / "logs",
        overwrite=True,
    )

    lines = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines()]
    assert stats == {"trajectories": 2, "memories": 2, "failed": 0, "dropped": 0}
    assert len(llm.calls) == 2
    assert [line["memory_id"] for line in lines] == ["m_000001", "m_000002"]
    assert lines[0]["source"]["trajectory_id"] == "traj_a"
    assert lines[1]["source"]["producer_agent"] == "multi-agent"
    assert set(lines[0]) == {
        "memory_id",
        "source",
        "source_task_description",
        "condition",
        "experience",
        "evidence",
    }
    assert "scope" not in lines[0]
    assert "agent_mask" not in lines[0]
    assert "target_agent" not in lines[0]


def test_run_extraction_attaches_source_metadata_from_mas_artifacts(tmp_path: Path):
    input_dir = tmp_path / "run"
    input_dir.mkdir()
    (input_dir / "trajectories.jsonl").write_text(
        json.dumps(
            {
                "run_id": "run_hotpot",
                "example_id": "hotpot_1",
                "dataset_name": "hotpotqa",
                "mas_type": "macnet",
                "messages": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (input_dir / "examples.jsonl").write_text(
        json.dumps(
            {
                "example_id": "hotpot_1",
                "dataset_name": "hotpotqa",
                "task_type": "qa",
                "input": {"question": "Who held the role?"},
                "target": {"answer": "Chief of Protocol"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (input_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "run_id": "run_hotpot",
                "example_id": "hotpot_1",
                "prediction": "ambassador",
                "target": {"answer": "Chief of Protocol"},
                "metrics": {"exact_match": 0.0, "token_f1": 0.0},
                "success": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    llm = FakeLLM([_memory_response(source={"producer_agent": "multi-agent", "producer_role": "team"})])
    output_file = tmp_path / "memories.jsonl"

    run_extraction(input_dir=input_dir, output_file=output_file, llm=llm, logs_dir=tmp_path / "logs", overwrite=True)

    row = json.loads(output_file.read_text(encoding="utf-8"))
    assert row["source"]["example_id"] == "hotpot_1"
    assert row["source"]["dataset_name"] == "hotpotqa"
    assert row["source"]["mas_type"] == "macnet"
    assert row["source"]["success"] is False
    assert row["source"]["metrics"] == {"exact_match": 0.0, "token_f1": 0.0}


def test_concrete_agent_team_role_is_normalized(tmp_path: Path):
    input_dir = tmp_path / "trajectories"
    input_dir.mkdir()
    (input_dir / "one.txt").write_text("trajectory", encoding="utf-8")
    llm = FakeLLM([_memory_response(source={"producer_agent": "summarizer agent", "producer_role": "team"})])
    output_file = tmp_path / "memories.jsonl"

    run_extraction(
        input_dir=input_dir,
        output_file=output_file,
        llm=llm,
        logs_dir=tmp_path / "logs",
        overwrite=True,
    )

    row = json.loads(output_file.read_text(encoding="utf-8"))
    assert row["source"]["producer_agent"] == "summarizer agent"
    assert row["source"]["producer_role"] == "summarizer agent"


def test_failed_response_is_logged_and_processing_continues(tmp_path: Path):
    input_dir = tmp_path / "trajectories"
    input_dir.mkdir()
    (input_dir / "a.jsonl").write_text(
        json.dumps({"trajectory_id": "bad", "messages": []})
        + "\n"
        + json.dumps({"trajectory_id": "good", "messages": []})
        + "\n",
        encoding="utf-8",
    )
    llm = FakeLLM(["not json", _memory_response()])
    output_file = tmp_path / "memories.jsonl"
    logs_dir = tmp_path / "logs"

    stats = run_extraction(
        input_dir=input_dir,
        output_file=output_file,
        llm=llm,
        logs_dir=logs_dir,
        overwrite=True,
    )

    lines = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines()]
    failed_logs = list(logs_dir.glob("*.json"))
    assert stats["failed"] == 1
    assert stats["memories"] == 1
    assert lines[0]["memory_id"] == "m_000001"
    assert lines[0]["source"]["trajectory_id"] == "good"
    assert len(failed_logs) == 1
    failed_payload = json.loads(failed_logs[0].read_text(encoding="utf-8"))
    assert failed_payload["trajectory_id"] == "bad"
    assert failed_payload["raw_response"] == "not json"


def test_request_failure_logs_empty_raw_response(tmp_path: Path):
    input_dir = tmp_path / "trajectories"
    input_dir.mkdir()
    (input_dir / "one.txt").write_text("trajectory", encoding="utf-8")
    output_file = tmp_path / "memories.jsonl"
    logs_dir = tmp_path / "logs"

    stats = run_extraction(
        input_dir=input_dir,
        output_file=output_file,
        llm=RaisingLLM(),
        logs_dir=logs_dir,
        overwrite=True,
    )

    failed_payload = json.loads(next(logs_dir.glob("*.json")).read_text(encoding="utf-8"))
    assert stats["failed"] == 1
    assert failed_payload["raw_response"] == ""


def test_items_missing_required_fields_are_dropped(tmp_path: Path):
    input_dir = tmp_path / "trajectories"
    input_dir.mkdir()
    (input_dir / "one.txt").write_text("trajectory", encoding="utf-8")
    llm = FakeLLM([json.dumps([{"source": {"producer_agent": "x"}}])])
    output_file = tmp_path / "memories.jsonl"

    stats = run_extraction(
        input_dir=input_dir,
        output_file=output_file,
        llm=llm,
        logs_dir=tmp_path / "logs",
        overwrite=True,
    )

    assert stats["dropped"] == 1
    assert output_file.read_text(encoding="utf-8") == ""
