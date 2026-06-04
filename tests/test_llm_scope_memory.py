import json

from mas_scope.config.schema import DatasetConfig, ExperimentConfig, LLMConfig, MASConfig, MemoryConfig, OutputConfig, TaskConfig
from mas_scope.core.registry import registry
from mas_scope.core.types import AgentSpec
from mas_scope.execution.runner import ExperimentRunner
from mas_scope.memory.bank import build_memory_bank
from mas_scope.memory.llm_scope_memory import LLMScopeMemoryProvider


def _write_memory_bank(path, records):
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def _memory_record(memory_id="m1", role="assistant agent", example_id="seed_1", dataset_name="hotpotqa"):
    return {
        "memory_id": memory_id,
        "source_example_id": example_id,
        "dataset_name": dataset_name,
        "task_type": "qa",
        "c_src": "seed question",
        "a_src": role,
        "r_src": role,
        "o_src": "Final Answer: Paris",
        "y_src": {"success": True, "metrics": {"exact_match": 1.0}},
        "p": {"reliability": 1.0, "access": "shared", "environment": None},
        "text": "Source role: assistant agent\nObserved output: Final Answer: Paris",
    }


def test_config_parses_llm_scope_memory(tmp_path):
    config = ExperimentConfig(
        experiment_name="memory_config",
        dataset=DatasetConfig(builder="hotpotqa", data_path="data/samples/hotpotqa_sample.json", limit=1),
        task=TaskConfig(type="qa"),
        mas=MASConfig(type="autogen"),
        llm=LLMConfig(provider="mock"),
        memory=MemoryConfig(provider="llm-scope", config={"bank_path": str(tmp_path / "bank.jsonl"), "policy_mode": "all-agents"}),
    )

    assert config.memory.provider == "llm-scope"
    assert config.memory.config["policy_mode"] == "all-agents"
    assert registry.get_memory_provider("llm-scope") is LLMScopeMemoryProvider


def test_deterministic_agent_mask_modes(tmp_path):
    bank_path = tmp_path / "bank.jsonl"
    _write_memory_bank(
        bank_path,
        [
            _memory_record("assistant-memory", "assistant agent"),
            _memory_record("proxy-memory", "user proxy agent"),
        ],
    )
    example = registry.get_dataset_builder("hotpotqa")("data/samples/hotpotqa_sample.json").build("dev", limit=1)[0]
    agent_spec = AgentSpec(name="assistant agent", role="assistant agent", system_prompt="")
    context = {"agent_order": ["assistant agent", "user proxy agent"], "current_agent": "assistant agent"}

    all_agents = LLMScopeMemoryProvider(bank_path=bank_path, policy_mode="all-agents")
    assert {memory["memory_id"] for memory in all_agents.retrieve(example, agent_spec, context)} == {"assistant-memory", "proxy-memory"}

    source_role = LLMScopeMemoryProvider(bank_path=bank_path, policy_mode="source-role-match")
    assert [memory["memory_id"] for memory in source_role.retrieve(example, agent_spec, context)] == ["assistant-memory"]

    assistant_only = LLMScopeMemoryProvider(bank_path=bank_path, policy_mode="assistant-only")
    assert {memory["memory_id"] for memory in assistant_only.retrieve(example, agent_spec, context)} == {"assistant-memory", "proxy-memory"}

    proxy_spec = AgentSpec(name="user proxy agent", role="user proxy agent", system_prompt="")
    proxy_context = {"agent_order": ["assistant agent", "user proxy agent"], "current_agent": "user proxy agent"}
    assert assistant_only.retrieve(example, proxy_spec, proxy_context) == []


def test_llm_mask_invalid_json_falls_back_to_empty(tmp_path):
    bank_path = tmp_path / "bank.jsonl"
    _write_memory_bank(bank_path, [_memory_record()])
    example = registry.get_dataset_builder("hotpotqa")("data/samples/hotpotqa_sample.json").build("dev", limit=1)[0]
    provider = LLMScopeMemoryProvider(
        bank_path=bank_path,
        policy_mode="llm-mask",
        policy_llm={"provider": "mock", "model": "mock-llm"},
    )

    memories = provider.retrieve(
        example,
        AgentSpec(name="assistant agent", role="assistant agent", system_prompt=""),
        {"agent_order": ["assistant agent", "user proxy agent"], "current_agent": "assistant agent"},
    )

    assert memories == []
    assert provider.last_retrieval_metadata["errors"]


def test_build_memory_bank_from_run_artifacts(tmp_path):
    run_dir = ExperimentRunner(
        ExperimentConfig(
            experiment_name="bank_source",
            dataset=DatasetConfig(builder="hotpotqa", data_path="data/samples/hotpotqa_sample.json", split="dev", limit=1),
            task=TaskConfig(type="qa"),
            mas=MASConfig(type="autogen"),
            llm=LLMConfig(provider="mock", model="mock-llm"),
            memory=MemoryConfig(provider="null"),
            output=OutputConfig(dir=tmp_path / "runs"),
        )
    ).run()
    bank_path = tmp_path / "memory_bank.jsonl"

    count = build_memory_bank(run_dir, bank_path)
    records = [json.loads(line) for line in bank_path.read_text(encoding="utf-8").splitlines()]

    assert count == 2
    assert records[0]["memory_id"]
    assert records[0]["source_example_id"]
    assert records[0]["task_type"] == "qa"
    assert records[0]["r_src"]
    assert records[0]["y_src"]["success"] is True
    assert records[0]["text"]


def test_runner_injects_assistant_only_memory_metadata(tmp_path):
    bank_path = tmp_path / "bank.jsonl"
    _write_memory_bank(bank_path, [_memory_record("seed-memory", "assistant agent", example_id="different")])
    config = ExperimentConfig(
        experiment_name="assistant_only_memory",
        dataset=DatasetConfig(builder="hotpotqa", data_path="data/samples/hotpotqa_sample.json", split="dev", limit=1),
        task=TaskConfig(type="qa"),
        mas=MASConfig(type="autogen"),
        llm=LLMConfig(provider="mock", model="mock-llm"),
        memory=MemoryConfig(
            provider="llm-scope",
            config={"bank_path": str(bank_path), "policy_mode": "assistant-only", "candidate_top_k": 5, "selected_top_k": 3},
        ),
        output=OutputConfig(dir=tmp_path / "runs"),
    )

    run_dir = ExperimentRunner(config).run()
    trajectory = json.loads((run_dir / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assistant_msg, user_proxy_msg = trajectory["messages"]

    assert assistant_msg["metadata"]["memory"]["selected_ids"] == ["seed-memory"]
    assert user_proxy_msg["metadata"]["memory"]["selected_ids"] == []
    assert assistant_msg["metadata"]["memory"]["agent_mask_rows"] == [[1, 0]]
