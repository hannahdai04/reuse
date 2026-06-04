# Code Documentation

## 1. Project Overview

This is Phase-1 infrastructure for an LLM-based multi-agent systems project studying cross-task experience memory reuse scope.

The implemented system can run no-memory MAS baselines over normalized benchmark examples. It includes dataset builders, task classes, LLM providers, MAS backbones, a runner, metrics, and artifact logging.

Not implemented:

- Real memory reuse.
- Learned memory scope.
- All-shared, all-private, or fixed-role memory.
- G-Memory, LatentMem, or LEGOMem memory logic.
- Bundled ALFWorld or ScienceWorld packages/data.

## 2. Repository Structure

- `configs/experiments/`: YAML configs for smoke and tiny real-file runs.
- `configs/environments/`: optional real environment config templates.
- `data/samples/`: tiny offline sample datasets.
- `data/real_samples/`: tiny real-file samples and manifests.
- `src/mas_scope/config/`: pydantic config schema and YAML loader.
- `src/mas_scope/core/`: shared schemas, registry, IDs, exceptions.
- `src/mas_scope/datasets/`: legacy adapters plus split-aware builders.
- `src/mas_scope/environments/`: environment adapter interface, mocks, optional real ALFWorld/ScienceWorld adapters.
- `src/mas_scope/evaluation/`: metrics for QA, planning, interactive tasks, and aggregation.
- `src/mas_scope/execution/`: experiment runner and artifact writer.
- `src/mas_scope/llm/`: base LLM interface, mock provider, OpenAI-compatible provider.
- `src/mas_scope/mas/`: MAS agent class, base interface, and four MAS backbones.
- `src/mas_scope/memory/`: memory interface and no-op provider.
- `src/mas_scope/prompts/`: centralized prompt templates.
- `src/mas_scope/tasks/`: task classes for QA, formal planning, and interactive episodes.
- `src/mas_scope/tools/`: normalization, action parsing, and PDDL format helpers.
- `tests/`: offline tests.

## 3. High-Level Data Flow

```text
Raw Dataset / Env Metadata
        |
        v
DatasetAdapter / DatasetBuilder
        |
        v
TaskExample
        |
        v
Task + MAS + LLM + MemoryProvider
        |
        v
Trajectory
        |
        v
Evaluation
        |
        v
Run Artifacts
```

The current runner uses `DatasetBuilder`, not the older `BaseDatasetAdapter`, for no-memory baseline experiments.

## 4. Core Schemas

All shared schemas live in `mas_scope.core.types`.

- `TaskExample`: normalized runnable example with `example_id`, `dataset_name`, `task_type`, `split`, `input`, `target`, and `metadata`.
- `AgentSpec`: task-facing description of an agent.
- `AgentMessage`: one MAS agent message with turn, role, content, and metadata.
- `ActionDecision`: interactive action plus messages and metadata.
- `EnvironmentState`: environment observation state.
- `EnvironmentStep`: logged environment transition.
- `Trajectory`: raw execution trace containing messages, environment steps, final answer/action, usage, and metadata.
- `TaskResult`: per-example prediction, target, metrics, success, cost, and error.
- `MessageGraph`: lightweight graph with state, messages, edges, action, and metadata.

Compatibility modules such as `datasets/schema.py`, `mas/messages.py`, and `environments/schema.py` re-export these schemas.

## 5. Dataset Adapters And Builders

`BaseDatasetAdapter` is a legacy parser interface with:

- `load(split, limit)`
- `validate_raw(example)`
- `normalize(raw, split)`

Existing adapters:

- `HotpotQAAdapter`
- `StrategyQAAdapter`
- `PDDLAdapter`

The no-memory baseline path uses `DatasetBuilder` in `datasets/builders.py`:

- `build(split, limit) -> list[TaskExample]`
- `available_splits() -> list[str]`

Registered builders:

- `HotpotQABuilder`: supports official HotpotQA list JSON, JSONL, and HF-style context/supporting facts dictionaries.
- `StrategyQABuilder`: supports StrategyQA JSON/JSONL and normalizes yes/no labels.
- `PDDLBuilder`: supports JSON/JSONL and IPC-style directories.
- `ALFWorldBuilder`: indexes `traj_data.json` files under ALFWorld split directories.
- `ScienceWorldBuilder`: reads manifest files containing task IDs/names and variation IDs.

Builders produce `TaskExample` objects only. They do not call LLMs, MAS, or environments.

## 6. Interactive Environment Adapters

`BaseEnvironmentAdapter` defines:

- `reset(example)`
- `step(action)`
- `close()`
- `get_action_space()`

Implemented mocks:

- `MockAlfworldEnvironment`
- `MockScienceWorldEnvironment`

Optional real adapters:

- `AlfworldEnvironment`
- `ScienceWorldEnvironment`

Real adapters import optional packages only in `__init__` and fail with clear errors if dependencies or required runtime config are missing.

`AlfworldEnvironment` follows the ALFWorld TextWorld API: `get_environment(env_type)(config, train_eval=...).init_env(batch_size=1)`, then batch `reset()` and `step([action])`. Split names are mapped as follows:

- `train` -> `train`
- `valid_seen`, `dev`, `validation` -> `eval_in_distribution`
- `valid_unseen`, `test` -> `eval_out_of_distribution`

The default real config is `configs/environments/alfworld_textworld_base.yaml`, which expects `ALFWORLD_DATA` to point to a directory containing `json_2.1.1/` and `logic/`.

`ScienceWorldEnvironment` follows the official example style: `ScienceWorldEnv("", jar_path, envStepLimit=...)`, then `load(task_name_or_id, variation_id, simplification, generateGoldPath)`, `reset()`, and `step(action)`. It obtains admissible actions from `info["valid"]`, `get_valid_action_object_combinations()`, or `get_valid_action_object_combinations_with_templates()`.

## 7. MAS Backbones

`BaseMAS` defines:

- `run(example, task, llm, memory) -> Trajectory`
- `act(example, observation, task, llm, memory, context) -> ActionDecision`

`MASAgent` stores:

- `role`
- `name`
- `id`
- `topology_node_id`
- `system_prompt_template`
- `user_prompt_template`

`MASAgent.invoke()` formats prompts, calls `BaseLLM.generate()`, and returns `AgentMessage`.

Registered MAS backbones:

- `AutoGenStyleMAS`: assistant agent -> user proxy agent.
- `MacNetStyleMAS`: actor/critic pairs -> summarizer.
- `CAMELStyleMAS`: assistant -> user -> assistant -> finalizer.
- `DyLANStyleMAS`: planner -> reasoner -> verifier -> optional reasoner revision -> finalizer.

All four use `MessageGraph`, `agents_list`, `topology`, and `mas_config`.

## 8. Task Classes

`BaseTask` defines:

- `build_prompt(example, agent_spec, context)`
- `parse_final_answer(trajectory)`
- `evaluate(prediction, example)`

Implemented tasks:

- `QATask`: builds QA prompts and computes QA metrics.
- `FormalPlanningTask`: builds PDDL planning prompts and computes lightweight plan metrics.
- `InteractiveTask`: owns the environment loop and calls `mas.act(...)`.

Interactive environments are controlled by `InteractiveTask`, not MAS classes.

`FormalPlanningTask` provides the domain PDDL, problem PDDL, allowed action
signatures, and declared objects/constants to the MAS. Its output contract asks
for grounded plan actions only: one complete parenthesized action per line, no
PDDL domain/problem files, no action schemas, no markdown, no variables such as
`?x`, and no explanations. `parse_final_answer()` uses
`tools.pddl_validator.clean_plan_output()` to extract complete action lines from
the model response before metric computation. The raw response remains in
`Trajectory.messages`.

Planning metrics are deliberately lightweight in Phase 1: `non_empty_plan`,
`valid_action_format`, `no_variables`, `uses_known_actions`,
`uses_declared_objects`, `plan_length`, and `exact_plan_match` when a reference
plan is present. They do not prove plan executability.

## 9. LLM Providers

`BaseLLM.generate(messages, **kwargs) -> LLMResponse`.

Providers:

- `MockLLM`: deterministic, offline, used in tests and smoke runs.
- `OpenAICompatibleLLM`: optional chat-completions-compatible HTTP provider using `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL`. The provider loads `.env` automatically through `core.env`; already-set shell variables take precedence.

MAS classes do not import model SDKs, torch, transformers, or external MAS frameworks.

## 10. Memory Interface

`MemoryProvider` defines:

- `retrieve(example, agent_spec, context) -> list[dict]`
- `update(example, trajectory, result) -> None`

`NullMemoryProvider`:

- `retrieve()` returns `[]`
- `update()` does nothing

Memory providers are registered through `registry.register_memory_provider(...)`.
Phase 1 registers only `"null"`, which resolves to `NullMemoryProvider`. The
runner creates memory through `registry.get_memory_provider(config.memory.provider)`.
This preserves the future extension point without implementing memory logic.

## 11. Runner And Artifacts

`ExperimentRunner`:

1. Loads an `ExperimentConfig`.
2. Creates a registered dataset builder.
3. Builds examples for a split.
4. Creates task, MAS, LLM, and a registered memory provider.
5. Runs each example.
6. Evaluates predictions.
7. Writes artifacts.

Artifact files:

- `config.yaml`: resolved config.
- `manifest.json`: run metadata.
- `examples.jsonl`: normalized `TaskExample` records.
- `trajectories.jsonl`: MAS trajectories.
- `environment_steps.jsonl`: interactive transitions.
- `results.jsonl`: per-example predictions and metrics.
- `metrics.json`: aggregate metrics.
- `errors.jsonl`: per-example errors.

## 12. CLI

Implemented commands:

```powershell
python -m mas_scope.cli validate-data --builder hotpotqa --data-path data/samples/hotpotqa_sample.json --split dev --limit 2
python -m mas_scope.cli list-splits --builder hotpotqa --data-path data/samples/hotpotqa_sample.json
python -m mas_scope.cli validate-env --environment alfworld-mock
python -m mas_scope.cli validate-env --environment scienceworld-mock
python -m mas_scope.cli run --config configs/experiments/smoke_hotpotqa_autogen.yaml
```

There is no `inspect-run` command yet.

Real interactive config templates:

```powershell
python -m mas_scope.cli run --config configs/experiments/alfworld_valid_seen_real3_autogen_openai.yaml
python -m mas_scope.cli run --config configs/experiments/alfworld_valid_unseen_real3_autogen_openai.yaml
python -m mas_scope.cli run --config configs/experiments/scienceworld_dev_real3_autogen_openai.yaml
```

## 13. Config Design

Config schema:

- `experiment_name`
- `dataset.builder`
- `dataset.data_path`
- `dataset.split`
- `dataset.limit`
- `task.type`
- `task.max_steps`
- `mas.type`
- `mas.config`
- `llm.provider`
- `llm.model`
- `llm.timeout`
- `llm.retries`
- `llm.temperature`
- `llm.max_tokens`
- `llm.extra_body`
- `memory.provider`
- `environment.provider`
- `environment.config`
- `output.dir`

Sample:

```yaml
experiment_name: smoke_hotpotqa_autogen
dataset:
  builder: hotpotqa
  data_path: data/samples/hotpotqa_sample.json
  split: dev
  limit: 2
task:
  type: qa
mas:
  type: autogen
llm:
  provider: mock
  model: mock-llm
  temperature: 0
  max_tokens: 128
  extra_body: {}
memory:
  provider: null
environment:
  provider: null
  config: {}
output:
  dir: runs
```

## 14. Adding A New Dataset

1. Implement a `DatasetBuilder` subclass.
2. Register it with `@registry.register_dataset_builder("name")`.
3. Implement `available_splits()` if file discovery is dataset-specific.
4. Implement `build(split, limit)` to return `TaskExample` objects.
5. Store runnable binding metadata such as `task_class`, `environment`, `source_path`, and dataset-specific IDs.
6. Add unit tests with tiny local files.

## 15. Adding A New Interactive Environment

1. Subclass `BaseEnvironmentAdapter`.
2. Implement `reset`, `step`, `close`, and `get_action_space`.
3. Add an environment name in runner environment creation.
4. Make optional heavy dependencies import inside `__init__`.
5. Add mock tests that do not require the real environment.

## 16. Adding A New MAS Backbone

1. Subclass `BaseMAS`.
2. Register with `@registry.register_mas("name")`.
3. Create `MASAgent` objects in `__init__`.
4. Maintain `self.agents_list`, `self.topology`, and `self.mas_config`.
5. Use `MessageGraph` to log messages and edges.
6. Return `Trajectory` from `run()` and `ActionDecision` from `act()`.
7. Add tests for construction, `run`, and `act`.

## 17. Adding Future Memory Methods

Future memory providers should subclass `MemoryProvider`.

Implementation steps:

1. Add a provider class under `src/mas_scope/memory/`.
2. Implement `retrieve()` and `update()`.
3. Register it with `@registry.register_memory_provider("name")`.
4. Select it with `memory.provider` in config.
5. Add tests showing behavior changes are isolated from dataset builders, tasks, and MAS interfaces.

Future examples:

- `AllSharedMemoryProvider`
- `AllPrivateMemoryProvider`
- `FixedRoleMemoryProvider`
- `LearnedScopeMemoryProvider`

These are not implemented in Phase 1.

## 18. Testing Strategy

Tests currently cover:

- QA prompt robustness.
- MAS registry and four MAS backbones.
- `run()` and `act()` for MAS classes.
- Dataset builder registry.
- HotpotQA original and HF-style shapes.
- StrategyQA boolean and string labels.
- PDDL JSONL and IPC-style directory layout.
- ALFWorld trial indexing.
- ScienceWorld manifest parsing.
- No-memory runner artifacts.
- CLI validation and run commands.

All tests run offline with `MockLLM`.

## 19. Common Failure Modes

- Dataset validation failure: check required fields for the selected builder and split.
- Empty trajectory: confirm the MAS class is registered and `run()` or `act()` returns messages.
- Interactive task never ends: lower `task.max_steps` and inspect `environment_steps.jsonl`.
- Missing environment dependency: real ALFWorld/ScienceWorld adapters are optional and raise clear errors.
- Missing ALFWorld data: set `ALFWORLD_DATA` to the directory that contains `json_2.1.1/` and `logic/`.
- Missing ScienceWorld jar: install `scienceworld`; set `SCIENCEWORLD_JAR_PATH` only if the package cannot find its bundled jar.
- OpenAI API error: check `.env` or the active shell values for `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL`.
- Invalid PDDL plan format: ensure generated actions are one parenthesized action per line.
- PDDL plan uses schema variables or unknown objects: check `Allowed action signatures` and `Declared objects/constants` in the prompt and inspect the raw trajectory message.
- YAML `provider: null`: accepted for memory and interpreted as no-memory.

## 20. Design Constraints

- Dataset builders and adapters must not call LLMs.
- Evaluation must not depend on MAS internals.
- MAS must not parse raw dataset formats.
- Environment adapters must expose `reset` and `step`.
- `InteractiveTask` owns the environment loop.
- `MAS.act(...)` only decides the next action.
- Tests must work offline.
- `MemoryProvider` must stay in runner signatures.
- Phase 1 must not implement real memory logic.
- Runner currently supports only the registered `null` memory provider.

## 21. Phase-1 Completion Checklist

The current codebase satisfies:

- `pytest` passes.
- Sample HotpotQA validates.
- Sample StrategyQA validates.
- Sample PDDL validates.
- No-memory HotpotQA smoke config runs.
- Real two-example PDDL sample validates and runs with mock/OpenAI-compatible providers.
- Run artifacts are written.
- `trajectories.jsonl` is written.
- `results.jsonl` and `metrics.json` are written.
- Dataset builders produce `TaskExample` objects.
- Mock environments support `reset` and `step`.
- `NullMemoryProvider` is registered and used.
- No real memory method is implemented.
- Real ALFWorld and ScienceWorld imports are optional.
- Tests do not require external APIs or heavy environment dependencies.

Still incomplete:

- Full benchmark configs for every dataset.
- `inspect-run` CLI.
- Full benchmark-scale real ALFWorld/ScienceWorld execution.
- Large-scale experiments.
