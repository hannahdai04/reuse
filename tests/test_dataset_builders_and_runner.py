import json
import subprocess
import sys

import yaml

from mas_scope.config.schema import DatasetConfig, ExperimentConfig, LLMConfig, MASConfig, MemoryConfig, OutputConfig, TaskConfig
from mas_scope.core.types import AgentSpec
from mas_scope.core.registry import registry
from mas_scope.execution.runner import ExperimentRunner
from mas_scope.tasks.formal_planning import FormalPlanningTask
from mas_scope.tools.pddl_validator import (
    clean_plan_output,
    extract_action_signatures,
    extract_declared_objects,
    plan_uses_declared_objects,
    plan_uses_known_actions,
)


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_dataset_builder_registry_lookup():
    assert registry.get_dataset_builder("hotpotqa").__name__ == "HotpotQABuilder"
    assert registry.get_dataset_builder("strategyqa").__name__ == "StrategyQABuilder"
    assert registry.get_dataset_builder("pddl").__name__ == "PDDLBuilder"
    assert registry.get_dataset_builder("alfworld").__name__ == "ALFWorldBuilder"
    assert registry.get_dataset_builder("scienceworld").__name__ == "ScienceWorldBuilder"


def test_memory_provider_registry_lookup():
    assert registry.get_memory_provider("null").__name__ == "NullMemoryProvider"
    assert registry.get_memory_provider("llm-scope").__name__ == "LLMScopeMemoryProvider"
    assert registry.get_memory_provider(None).__name__ == "NullMemoryProvider"
    assert registry.memory_provider_names() == ["llm-scope", "null"]


def test_hotpotqa_builder_original_and_hf_shapes(tmp_path):
    original = tmp_path / "hotpot.json"
    write_json(
        original,
        [
            {
                "_id": "h1",
                "question": "Where is the Eiffel Tower?",
                "answer": "Paris",
                "supporting_facts": [["Eiffel Tower", 0]],
                "context": [["Eiffel Tower", ["The Eiffel Tower is in Paris."]]],
            }
        ],
    )
    examples = registry.get_dataset_builder("hotpotqa")(original).build("dev")
    assert examples[0].example_id == "h1"
    assert examples[0].target["answer"] == "Paris"
    assert examples[0].metadata["task_class"] == "qa"

    hf = tmp_path / "hotpot_hf.json"
    write_json(
        hf,
        [
            {
                "id": "h2",
                "question": "Where is Paris?",
                "answer": "France",
                "supporting_facts": {"title": ["Paris"], "sent_id": [0]},
                "context": {"title": ["Paris"], "sentences": [["Paris is in France."]]},
            }
        ],
    )
    hf_examples = registry.get_dataset_builder("hotpotqa")(hf).build("validation")
    assert hf_examples[0].input["context"] == [["Paris", ["Paris is in France."]]]
    assert hf_examples[0].target["supporting_facts"] == [["Paris", 0]]


def test_hotpotqa_unlabeled_test_does_not_require_answer(tmp_path):
    path = tmp_path / "hotpot_test.json"
    write_json(path, [{"_id": "test1", "question": "Unlabeled?", "context": []}])
    example = registry.get_dataset_builder("hotpotqa")(path).build("test")[0]
    assert example.target == {}


def test_strategyqa_builder_boolean_and_yes_no_answers(tmp_path):
    path = tmp_path / "strategy.json"
    write_json(
        path,
        [
            {"qid": "s1", "question": "Can penguins fly?", "answer": False, "facts": ["Penguins are flightless."]},
            {"qid": "s2", "question": "Is water wet?", "answer": "yes", "decomposition": ["What is water?"]},
        ],
    )
    examples = registry.get_dataset_builder("strategyqa")(path).build("dev")
    assert examples[0].target["normalized_answer"] == "no"
    assert examples[1].target["normalized_answer"] == "yes"
    assert examples[0].input["facts"] == ["Penguins are flightless."]


def test_pddl_builder_jsonl_and_ipc_directory(tmp_path):
    jsonl = tmp_path / "pddl.jsonl"
    jsonl.write_text(
        '{"id":"p1","instruction":"Move.","domain_pddl":"(define (domain d))","problem_pddl":"(define (problem p) (:goal (done)))","reference_plan":["(move a b)"]}\n',
        encoding="utf-8",
    )
    examples = registry.get_dataset_builder("pddl")(jsonl).build("test")
    assert examples[0].task_type == "formal_planning"
    assert examples[0].target["reference_plan"] == ["(move a b)"]

    ipc = tmp_path / "blocksworld"
    (ipc / "instances").mkdir(parents=True)
    (ipc / "domain.pddl").write_text("(define (domain blocks))", encoding="utf-8")
    (ipc / "instances" / "instance-1.pddl").write_text("(define (problem p1) (:goal (clear a)))", encoding="utf-8")
    ipc_examples = registry.get_dataset_builder("pddl")(ipc).build("default")
    assert ipc_examples[0].metadata["benchmark_format"] == "ipc-directory"
    assert "domain_pddl" in ipc_examples[0].input


def test_formal_planning_prompt_rejects_schema_output():
    example = registry.get_dataset_builder("pddl")("data/real_samples/pddl/ipc-2000/blocks-strips-typed").build("default", limit=1)[0]
    prompt = FormalPlanningTask().build_prompt(
        example,
        agent_spec=AgentSpec(name="Planner", role="planner", system_prompt="Plan."),
        context={},
    )

    assert "Allowed action signatures" in prompt
    assert "Declared objects/constants" in prompt
    assert "Do not output PDDL schema sections" in prompt
    assert "grounded plan actions" in prompt
    assert "Never use variables" in prompt


def test_pddl_prompt_helpers_extract_constraints():
    domain_pddl = """
    (define (domain move-domain)
      (:constants depot - location)
      (:action move
        :parameters (?from ?to - location)
        :precondition (and)
        :effect (and))
    )
    """
    problem_pddl = """
    (define (problem move-problem)
      (:domain move-domain)
      (:objects room-a room-b - location)
      (:goal (at room-b))
    )
    """

    assert extract_action_signatures(domain_pddl) == [{"name": "move", "parameters": ["?from", "?to"], "arity": 2}]
    assert extract_declared_objects(problem_pddl, domain_pddl) == ["depot", "room-a", "room-b"]
    assert plan_uses_known_actions("(move room-a room-b)", ["move"])
    assert plan_uses_declared_objects("(move room-a room-b)", ["room-a", "room-b"])
    assert not plan_uses_declared_objects("(move room-a room-c)", ["room-a", "room-b"])


def test_pddl_plan_output_cleaning_extracts_complete_actions():
    raw = """
    1. (pick-up A)
    (stack A B)
    Final Answer: (stack A B)
    Final Answer: (stack B C)
    (stack C
    """

    assert clean_plan_output(raw) == "(pick-up A)\n(stack A B)\n(stack B C)"


def test_alfworld_builder_indexes_trial_directory(tmp_path):
    trial = tmp_path / "json_2.1.1" / "train" / "pick_and_place-Apple-None-Fridge-1" / "trial_T1"
    trial.mkdir(parents=True)
    write_json(
        trial / "traj_data.json",
        {"task_id": "alf_1", "task_type": "pick_and_place", "turk_annotations": {"anns": [{"task_desc": "put apple in fridge"}]}},
    )
    (trial / "game.tw-pddl").write_text("game", encoding="utf-8")
    (trial / "initial_state.pddl").write_text("(init)", encoding="utf-8")

    examples = registry.get_dataset_builder("alfworld")(tmp_path).build("train")
    assert examples[0].example_id == "alf_1"
    assert examples[0].metadata["environment"] == "alfworld"
    assert examples[0].input["instruction"] == "put apple in fridge"


def test_scienceworld_builder_uses_manifest_without_importing_env(tmp_path):
    manifest = tmp_path / "scienceworld.json"
    write_json(
        manifest,
        [{"id": "sw1", "split": "dev", "task_id": "2-1", "task_name": "use-thermometer", "variation_id": 0}],
    )
    examples = registry.get_dataset_builder("scienceworld")(manifest).build("dev")
    assert examples[0].metadata["environment"] == "scienceworld"
    assert examples[0].input["task_name"] == "use-thermometer"


def make_config(tmp_path, mas_type="autogen"):
    return ExperimentConfig(
        experiment_name=f"runner_{mas_type}",
        dataset=DatasetConfig(builder="hotpotqa", data_path="data/samples/hotpotqa_sample.json", split="dev", limit=1),
        task=TaskConfig(type="qa"),
        mas=MASConfig(type=mas_type),
        llm=LLMConfig(provider="mock", model="mock-llm"),
        memory=MemoryConfig(provider="null"),
        output=OutputConfig(dir=tmp_path / "runs"),
    )


def test_no_memory_runner_with_all_mas_backbones_on_qa(tmp_path):
    for mas_type in ("autogen", "macnet", "camel", "dylan"):
        run_dir = ExperimentRunner(make_config(tmp_path, mas_type)).run()
        assert (run_dir / "examples.jsonl").exists()
        assert (run_dir / "trajectories.jsonl").read_text(encoding="utf-8").strip()
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["memory_provider"] == "null"
        results = [json.loads(line) for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]
        assert results[0]["prediction"] == "Paris"
        assert results[0]["metrics"]["exact_match"] == 1.0


def test_cli_validate_data_and_run(tmp_path):
    validate = subprocess.run(
        [
            sys.executable,
            "-m",
            "mas_scope.cli",
            "validate-data",
            "--builder",
            "hotpotqa",
            "--data-path",
            "data/samples/hotpotqa_sample.json",
            "--split",
            "dev",
            "--limit",
            "2",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "validated 2 examples" in validate.stdout

    config_path = tmp_path / "smoke.yaml"
    config_payload = make_config(tmp_path).model_dump(mode="json")
    config_path.write_text(yaml.safe_dump(config_payload), encoding="utf-8")
    run = subprocess.run(
        [sys.executable, "-m", "mas_scope.cli", "run", "--config", str(config_path)],
        text=True,
        capture_output=True,
        check=True,
    )
    run_dir = run.stdout.strip()
    assert run_dir


def test_runner_sanitizes_model_name_in_run_dir(tmp_path):
    config = make_config(tmp_path)
    config.llm.model = "provider/model name"

    run_dir = ExperimentRunner(config).run()

    assert run_dir.parent == tmp_path / "runs"
    assert "provider_model_name" in run_dir.name
