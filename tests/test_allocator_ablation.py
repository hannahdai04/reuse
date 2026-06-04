import json
from pathlib import Path

from allocator.diagnostics import compute_diagnostics
from allocator.masking import generate_mask
from allocator.realization import realize_memories
from allocator.retrieval import load_memories, retrieve_top_k
from allocator.runtime_provider import apply_allowed_cap
from allocator.selection import select_memories
from mas_scope.core.types import TaskExample
from mas_scope.llm.mock import MockLLM
from scripts.analyze_allocator_ablation import main as analyze_main
from scripts.generate_allocator_report import main as report_main
from scripts.run_allocator_ablation import main as run_main


def _example(example_id="eval_1"):
    return TaskExample(
        example_id=example_id,
        dataset_name="hotpotqa",
        task_type="qa",
        split="dev",
        input={
            "question": "Where is the Eiffel Tower?",
            "context": [["Eiffel Tower", ["The Eiffel Tower is in Paris."]]],
        },
        target={"answer": "Paris"},
    )


def _memory(memory_id="m_000001", task="HotpotQA"):
    return {
        "memory_id": memory_id,
        "source": {"trajectory_id": "seed", "producer_agent": "multi-agent", "producer_role": "team"},
        "source_task_description": f"{task} task with weak critique.",
        "condition": "When critic agents simply repeat actor answers.",
        "experience": "Critics should verify actor answers instead of repeating them.",
        "evidence": "The source trajectory succeeded but showed weak critique.",
    }


def _write_jsonl(path: Path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_memory_loader_namespaces_duplicate_ids(tmp_path: Path):
    strategy = tmp_path / "strategyqa_memories.jsonl"
    hotpot = tmp_path / "hotpotqa_memories.jsonl"
    _write_jsonl(strategy, [_memory("m_000001", "StrategyQA")])
    _write_jsonl(hotpot, [_memory("m_000001", "HotpotQA")])

    memories = load_memories([strategy, hotpot])

    assert {memory["memory_id"] for memory in memories} == {"strategyqa:m_000001", "hotpotqa:m_000001"}
    assert {memory["original_memory_id"] for memory in memories} == {"m_000001"}


def test_retrieval_is_deterministic(tmp_path: Path):
    memory_file = tmp_path / "hotpotqa_memories.jsonl"
    _write_jsonl(
        memory_file,
        [
            _memory("m_1", "HotpotQA"),
            {**_memory("m_2", "HotpotQA"), "experience": "Use retrieval for football team history."},
        ],
    )
    memories = load_memories([memory_file])

    first = retrieve_top_k(_example(), memories, top_k=2)
    second = retrieve_top_k(_example(), memories, top_k=2)

    assert [item["memory_id"] for item in first] == [item["memory_id"] for item in second]
    assert [item["retrieval_rank"] for item in first] == [1, 2]


def test_masking_invalid_json_falls_back_to_all_zero():
    result = generate_mask(MockLLM(), _example(), [_memory("hotpotqa:m1")], ["actor", "critic"])

    assert result["mask_matrix"] == [[0, 0]]
    assert result["errors"]
    assert result["mask_reasons"][0]["reason_tag"] == "allocator_error"


def test_selection_enforces_budget_with_fallback():
    memories = [
        {"memory_id": "m1", "retrieval_rank": 1, "retrieval_score": 3},
        {"memory_id": "m2", "retrieval_rank": 2, "retrieval_score": 2},
        {"memory_id": "m3", "retrieval_rank": 3, "retrieval_score": 1},
    ]

    result = select_memories(MockLLM(), _example(), memories, ["actor"], [[1], [1], [1]], per_agent_k=2, use_llm=False)

    assert result["selected_matrix"] == [[1], [1], [0]]


def test_allowed_cap_limits_each_agent_by_score():
    memories = [
        {"memory_id": "m1", "retrieval_rank": 1},
        {"memory_id": "m2", "retrieval_rank": 2},
        {"memory_id": "m3", "retrieval_rank": 3},
    ]
    result = apply_allowed_cap(
        memories,
        ["actor", "critic"],
        [[1, 1], [1, 1], [1, 1]],
        cap=1,
        agent_score_matrix=[[0.1, 0.9], [0.8, 0.2], [0.5, 0.7]],
    )

    assert result["mask_matrix"] == [[0, 1], [1, 0], [0, 0]]
    assert result["cap_stats"]["dropped_count"] == 4
    assert result["cap_stats"]["avg_allowed_after_cap"] == 1.0


def test_realization_invalid_json_falls_back_to_raw():
    memories = [{"memory_id": "m1", "text": "Condition: x\nExperience: y", "retrieval_rank": 1}]

    result = realize_memories(MockLLM(), _example(), memories, ["actor"], [[1]], use_llm=True)

    assert result["errors"]
    assert result["realized_memories"]["actor"][0]["realization_type"] == "raw"
    assert result["realized_memories"]["actor"][0]["text"] == "Condition: x\nExperience: y"


def test_diagnostics_computes_expected_rates():
    records = [
        {
            "retrieved_memories": [{"memory_id": "m1"}, {"memory_id": "m2"}],
            "agents": ["a1", "a2"],
            "mask_matrix": [[1, 0], [1, 1]],
            "mask_reasons": [{"reason_tag": "critic"}, {"reason_tag": "all_shared"}],
            "selected_matrix": [[1, 0], [0, 1]],
            "realized_memories": {
                "a1": [{"text": "abc def", "realization_type": "raw"}],
                "a2": [{"text": "abc xyz", "realization_type": "warning"}],
            },
            "task_result": {"success": True, "metrics": {"exact_match": 1.0}},
            "allocator_errors": [],
        }
    ]

    diagnostics = compute_diagnostics(records)

    assert diagnostics["retrieval"]["avg_retrieved_memory_count"] == 2
    assert diagnostics["masking"]["mask_density"] == 0.75
    assert diagnostics["task"]["success_rate"] == 1.0
    assert diagnostics["realization"]["realization_type_counts"] == {"raw": 1, "warning": 1}


def test_run_analyze_and_report_smoke(tmp_path: Path):
    target_file = tmp_path / "targets.jsonl"
    memory_file = tmp_path / "hotpotqa_memories.jsonl"
    _write_jsonl(target_file, [_example().model_dump(mode="json")])
    _write_jsonl(memory_file, [_memory("m_000001", "HotpotQA")])
    root = tmp_path / "ablation" / "hotpotqa"

    for setting in ("B0", "B1", "B2"):
        args = [
            "--target_file",
            str(target_file),
            "--setting",
            setting,
            "--task_provider",
            "mock",
            "--mas_model",
            "mock-llm",
            "--allocator_provider",
            "mock",
            "--allocator_model",
            "mock-llm",
            "--output_dir",
            str(root / setting),
            "--overwrite",
        ]
        if setting != "B0":
            args.extend(["--memory_file", str(memory_file)])
        run_main(args)

    b0_record = json.loads((root / "B0" / "allocation_records.jsonl").read_text(encoding="utf-8").splitlines()[0])
    b1_record = json.loads((root / "B1" / "allocation_records.jsonl").read_text(encoding="utf-8").splitlines()[0])
    b2_record = json.loads((root / "B2" / "allocation_records.jsonl").read_text(encoding="utf-8").splitlines()[0])

    assert b0_record["retrieved_memories"] == []
    assert b0_record["allocator_errors"] == []
    assert b1_record["retrieved_memories"][0]["memory_id"] == "hotpotqa:m_000001"
    assert b1_record["mask_matrix"][0] == [1, 1, 1, 1, 1]
    assert b2_record["mask_matrix"][0] == [0, 0, 0, 0, 0]

    run_main(
        [
            "--target_file",
            str(target_file),
            "--setting",
            "G2",
            "--task_provider",
            "mock",
            "--mas_model",
            "mock-llm",
            "--allocator_provider",
            "mock",
            "--allocator_model",
            "mock-llm",
            "--memory_file",
            str(memory_file),
            "--global_chunk_size",
            "1",
            "--mask_allowed_per_agent_k",
            "1",
            "--output_dir",
            str(root / "G2"),
            "--overwrite",
        ]
    )
    g2_record = json.loads((root / "G2" / "allocation_records.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert g2_record["global_masking"] is True
    assert g2_record["allowed_cap_stats"]["enabled"] is True

    summary = tmp_path / "ablation" / "summary.csv"
    report = tmp_path / "ablation" / "analysis_report.md"
    analyze_main(["--input_dir", str(tmp_path / "ablation"), "--output_file", str(summary)])
    report_main(["--input_dir", str(tmp_path / "ablation"), "--summary_file", str(summary), "--output_file", str(report)])

    assert summary.exists()
    assert "B0-B4 Comparison" in report.read_text(encoding="utf-8")
