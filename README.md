# MAS Scope

Phase-1 infrastructure for studying cross-task experience memory reuse scope in LLM-based multi-agent systems.

Chinese developer documentation: [docs/CODE_DOCUMENTATION_ZH.md](docs/CODE_DOCUMENTATION_ZH.md).

The current codebase supports a no-memory baseline pipeline:

```text
DatasetBuilder -> TaskExample -> Task + MAS + LLM + NullMemoryProvider -> Trajectory -> Evaluation -> Run Artifacts
```

No real memory method is implemented. `NullMemoryProvider` is the only memory provider.

## Phase-1 Scope

Implemented:

- Split-aware dataset builders for HotpotQA, StrategyQA, PDDL, ALFWorld metadata, and ScienceWorld manifests.
- In-house MAS framework with registry, agent objects, topology, `MessageGraph`, and trajectory output.
- MAS backbones: AutoGen-style, MacNet-style, CAMEL-style, DyLAN-style.
- Mock LLM and optional OpenAI-compatible LLM provider.
- QA, formal planning, and interactive task classes.
- No-memory experiment runner.
- CLI for dataset validation, split listing, and smoke runs.
- Real-run config templates for ALFWorld TextWorld and ScienceWorld manifests.
- JSONL/JSON/YAML run artifacts.
- Offline pytest suite.

Intentionally not implemented:

- Real memory methods.
- Learned memory scope.
- All-shared, all-private, or fixed-role memory.
- G-Memory, LatentMem, or LEGOMem memory logic.
- Large-scale benchmark execution.
- Bundled ALFWorld or ScienceWorld installations/data.

## Installation

Python 3.11+ is required.

```powershell
cd D:\agent_memory_code\reuse
python -m pip install -e ".[dev]"
```

Run tests:

```powershell
python -m pytest
```

## Environment Variables

The test suite does not require external APIs.

For optional OpenAI-compatible inference:

```env
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=...
```

The CLI and `OpenAICompatibleLLM` load `.env` automatically. Shell environment variables still take precedence when they are already set.

For real ALFWorld:

```env
ALFWORLD_DATA=D:/path/to/alfworld/data
```

`ALFWORLD_DATA` must point to a directory containing `json_2.1.1/` and `logic/`.
The provided real config uses `configs/environments/alfworld_textworld_base.yaml`.

For real ScienceWorld, install the `scienceworld` package. If the package cannot
find its bundled jar, set:

```env
SCIENCEWORLD_JAR_PATH=D:/path/to/scienceworld.jar
```

## Repository Structure

```text
configs/experiments/        Smoke experiment configs
configs/environments/       Optional real environment config templates
data/samples/               Tiny offline sample datasets
data/real_samples/          Tiny real-file samples and manifests
src/mas_scope/config/       YAML config schema and loader
src/mas_scope/core/         Shared schemas, IDs, registries, exceptions
src/mas_scope/datasets/     Dataset adapters and split-aware builders
src/mas_scope/environments/ Environment adapter interfaces and mocks
src/mas_scope/evaluation/   QA, planning, interactive, aggregate metrics
src/mas_scope/execution/    No-memory runner and artifact writer
src/mas_scope/llm/          Base, mock, OpenAI-compatible LLM providers
src/mas_scope/mas/          In-house MAS framework and backbones
src/mas_scope/memory/       Memory interface and NullMemoryProvider
src/mas_scope/prompts/      MAS prompt templates
src/mas_scope/tasks/        QA, formal planning, interactive tasks
src/mas_scope/tools/        Text/action/PDDL utility helpers
tests/                      Offline unit and integration tests
```

## Dataset Builders

Builders are registered with:

```python
@registry.register_dataset_builder("hotpotqa")
class HotpotQABuilder(DatasetBuilder):
    ...
```

Available builders:

- `hotpotqa`: official JSON list, JSONL, and Hugging Face-style `context.title/context.sentences`.
- `strategyqa`: official JSON list and JSONL; boolean/string answers are normalized to `yes`/`no`.
- `pddl`: JSON/JSONL records or IPC-style directories with `domain.pddl` and `instances/*.pddl`.
- `alfworld`: indexes real ALFWorld `json_2.1.1/{split}/.../traj_data.json` trial directories.
- `scienceworld`: reads local manifests of task IDs/names and variation IDs.

Builders output `TaskExample` records. They do not call LLMs or MAS code.

## CLI

Validate data:

```powershell
python -m mas_scope.cli validate-data --builder hotpotqa --data-path data/samples/hotpotqa_sample.json --split dev --limit 2
python -m mas_scope.cli validate-data --builder strategyqa --data-path data/samples/strategyqa_sample.json --split dev --limit 2
python -m mas_scope.cli validate-data --builder pddl --data-path data/samples/pddl_sample.jsonl --split test --limit 1
python -m mas_scope.cli validate-data --builder scienceworld --data-path data/real_samples/scienceworld_manifest_small.json --split dev --limit 3
```

List splits:

```powershell
python -m mas_scope.cli list-splits --builder hotpotqa --data-path data/samples/hotpotqa_sample.json
```

Run a no-memory smoke baseline:

```powershell
python -m mas_scope.cli run --config configs/experiments/smoke_hotpotqa_autogen.yaml
```

Validate environment adapters:

```powershell
python -m mas_scope.cli validate-env --environment alfworld-mock
python -m mas_scope.cli validate-env --environment scienceworld-mock
```

Run the real two-example PDDL sample:

```powershell
python -m mas_scope.cli validate-data --builder pddl --data-path data/real_samples/pddl/ipc-2000/blocks-strips-typed --split default --limit 2
python -m mas_scope.cli run --config configs/experiments/pddl_real2_autogen_mock.yaml
python -m mas_scope.cli run --config configs/experiments/pddl_real2_autogen_openai.yaml
```

The PDDL prompt follows the plan-generation pattern used by planning benchmarks:
it gives the domain/problem PDDL, lists allowed action signatures and declared
objects, and asks for plan actions only. `FormalPlanningTask.parse_final_answer()`
then extracts complete parenthesized actions from the model response for
Phase-1 lightweight metrics. This is still not semantic plan validation; no
external planner is called.

Run real interactive small samples after installing the external environment
packages and setting required paths:

```powershell
python -m mas_scope.cli run --config configs/experiments/alfworld_valid_seen_real3_autogen_openai.yaml
python -m mas_scope.cli run --config configs/experiments/alfworld_valid_unseen_real3_autogen_openai.yaml
python -m mas_scope.cli run --config configs/experiments/scienceworld_dev_real3_autogen_openai.yaml
```

## Experiment Config

Example:

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

`memory.provider: null` is parsed as YAML `None`; the memory registry maps it to `NullMemoryProvider`.

## Run Artifacts

Each run writes:

```text
runs/{timestamp}_{dataset}_{mas}_{safe_model}/
  config.yaml
  manifest.json
  examples.jsonl
  trajectories.jsonl
  environment_steps.jsonl
  results.jsonl
  metrics.json
  errors.jsonl
```

`trajectories.jsonl` stores raw MAS messages and metadata. These trajectories are not memory; future memory methods can consume them later.

## Adding A Dataset

1. Add a builder in `src/mas_scope/datasets/builders.py` or a new dataset module.
2. Register it with `@registry.register_dataset_builder("name")`.
3. Implement `build(split, limit)` and `available_splits()`.
4. Return `TaskExample` objects with `metadata.task_class` and optional `metadata.environment`.
5. Add tests with tiny local files.

## Adding A MAS Backbone

1. Subclass `BaseMAS`.
2. Register it with `@registry.register_mas("name")`.
3. Create `MASAgent` objects in `__init__`.
4. Maintain `self.agents_list`, `self.topology`, and `self.mas_config`.
5. Use `MessageGraph` in `run()` and `act()`.

## Future Memory Methods

Future providers should subclass `MemoryProvider`, register with
`@registry.register_memory_provider("name")`, and implement:

- `retrieve(example, agent_spec, context)`
- `update(example, trajectory, result)`

Examples that may be added later:

- `AllSharedMemoryProvider`
- `AllPrivateMemoryProvider`
- `FixedRoleMemoryProvider`
- `LearnedScopeMemoryProvider`

They are not implemented in Phase 1.

The runner already creates memory providers through `registry.get_memory_provider(...)`,
so future memory methods should not require dataset, task, or MAS runner rewrites.

## Verified Commands

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python -m mas_scope.cli validate-data --builder hotpotqa --data-path data/samples/hotpotqa_sample.json --split dev --limit 2
python -m mas_scope.cli run --config configs/experiments/smoke_hotpotqa_autogen.yaml
```

Current test status:

```text
36 passed
```
