# 代码文档（中文）

## 1. 项目定位

本项目是 LLM-based multi-agent systems 记忆复用范围研究的 Phase 1 基础设施。当前目标是稳定 no-memory baseline，而不是实现记忆方法。

```text
DatasetBuilder -> TaskExample -> Task + MAS + LLM + NullMemoryProvider -> Trajectory -> Evaluation -> Run Artifacts
```

当前只实现 `NullMemoryProvider`。没有实现 learned memory scope、all-shared memory、all-private memory、fixed-role memory、G-Memory、LatentMem 或 LEGOMem。

## 2. 当前可运行能力

已经实现：

- HotpotQA、StrategyQA、PDDL、ALFWorld metadata、ScienceWorld manifest 的 dataset builder。
- 内部 MAS framework：registry、`MASAgent`、topology、`MessageGraph`、trajectory logging。
- 四个 MAS backbone：`autogen`、`macnet`、`camel`、`dylan`。
- `MockLLM` 和可选的 `OpenAICompatibleLLM`。
- `QATask`、`FormalPlanningTask`、`InteractiveTask`。
- no-memory `ExperimentRunner`。
- CLI：`validate-data`、`list-splits`、`run`。
- JSONL/JSON/YAML artifact 输出。
- 离线 pytest 测试。
- PDDL 真实 IPC 小样本验证。
- 真实 ALFWorld / ScienceWorld adapter 的基础包装。

当前没有实现：

- 真实 memory 方法。
- 大规模 benchmark 自动实验。
- 大规模真实 ALFWorld / ScienceWorld 实验。
- `inspect-run` CLI。

## 3. 目录结构

```text
configs/experiments/        实验 YAML 配置
data/samples/               离线小样本数据
data/real_samples/          下载的真实小样本数据
src/mas_scope/config/       配置 schema 与 YAML loader
src/mas_scope/core/         共享类型、registry、ID、环境变量加载、异常
src/mas_scope/datasets/     legacy dataset adapters 与 DatasetBuilder
src/mas_scope/environments/ 环境接口、mock 环境、真实 ALFWorld / ScienceWorld adapter
src/mas_scope/evaluation/   QA / planning / interactive / aggregate metrics
src/mas_scope/execution/    runner 与 artifact writer
src/mas_scope/llm/          LLM interface、MockLLM、OpenAI-compatible provider
src/mas_scope/mas/          MASAgent、BaseMAS、四个 MAS backbone
src/mas_scope/memory/       MemoryProvider 与 NullMemoryProvider
src/mas_scope/prompts/      MAS prompt templates
src/mas_scope/tasks/        QA、formal planning、interactive task
src/mas_scope/tools/        文本归一化、答案解析、action 解析、PDDL 校验
tests/                      离线测试
```

## 4. 核心数据流

```text
Raw Dataset / Env Metadata
        |
        v
DatasetBuilder
        |
        v
TaskExample
        |
        v
Task + MAS + LLM + NullMemoryProvider
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

当前 runner 使用 `DatasetBuilder`，不是 legacy `BaseDatasetAdapter`。legacy adapter 保留是为了接口兼容和轻量解析能力。

## 5. 共享 Schema

共享类型集中在 `src/mas_scope/core/types.py`：

- `TaskExample`：标准化后的可运行样本。
- `AgentSpec`：Task 侧看到的 agent 描述。
- `AgentMessage`：单个 agent 的输出消息。
- `ActionDecision`：interactive task 的 action 决策。
- `EnvironmentState`：环境 observation 状态。
- `EnvironmentStep`：环境 step 日志。
- `Trajectory`：完整 MAS 执行轨迹。
- `TaskResult`：单样本预测、target、metrics、success、cost、error。
- `MessageGraph`：轻量消息图，记录 messages、edges、action。

兼容模块 `datasets/schema.py`、`mas/messages.py`、`environments/schema.py` 仍然 re-export 这些类型。新代码应优先从 `mas_scope.core.types` 导入。

## 6. DatasetBuilder

builder 注册方式：

```python
@registry.register_dataset_builder("hotpotqa")
class HotpotQABuilder(DatasetBuilder):
    ...
```

现有 builder：

- `HotpotQABuilder`：支持官方 JSON list、JSONL、Hugging Face 风格 `context.title/context.sentences`。
- `StrategyQABuilder`：支持官方 JSON list / JSONL，把 bool/string answer 归一化成 `yes/no`。
- `PDDLBuilder`：支持 JSON/JSONL，也支持 IPC-style 目录结构。
- `ALFWorldBuilder`：索引 `json_2.1.1/{split}/.../traj_data.json`。
- `ScienceWorldBuilder`：读取本地 manifest，不在 builder 阶段 import ScienceWorld。

builder 只负责把原始数据变成 `TaskExample`。它不能调用 LLM、MAS 或真实环境。

## 7. MAS Framework

MAS 注册方式：

```python
@registry.register_mas("autogen")
class AutoGenStyleMAS(BaseMAS):
    ...
```

`BaseMAS` 保留两个入口：

- `run(example, task, llm, memory) -> Trajectory`：用于 QA / PDDL。
- `act(example, observation, task, llm, memory, context) -> ActionDecision`：用于 interactive task。

当前真实样本配置使用 `AutoGenStyleMAS`：

```text
assistant agent -> user proxy agent
```

## 8. QA Prompt 与答案解析

`QATask.build_prompt()` 要求最终输出一行：

```text
Final Answer: <answer>
```

HotpotQA 要求最小 span/entity；StrategyQA 要求 `yes/no`。答案解析在 `src/mas_scope/tools/answer_parser.py`，只清理格式噪声，不根据 gold 做语义猜测。

## 9. PDDL 适配

真实 PDDL 小样本来自 potassco `pddl-instances` 的 IPC 2000 blocks-strips-typed：

```text
data/real_samples/pddl/ipc-2000/blocks-strips-typed/
  domain.pddl
  instances/
    instance-1.pddl
    instance-2.pddl
```

配置：

- `configs/experiments/pddl_real2_autogen_mock.yaml`
- `configs/experiments/pddl_real2_autogen_openai.yaml`

`FormalPlanningTask` 的 prompt 明确要求：

- 这是 planning task，不是 domain modeling task。
- 不要输出 `:action`、`:parameters`、`:precondition`、`:effect`。
- 不要输出 PDDL domain file 或 PDDL problem file。
- 不要输出 markdown、代码块、编号、JSON 或 YAML。
- 只输出 grounded plan actions。
- 每行一个完整的 parenthesized action。
- 每个 action 必须使用 prompt 中列出的 action name 和 objects/constants。
- 不要在最终 plan 中使用 `?x`、`?from`、`?to` 这类 schema variables。

`FormalPlanningTask.parse_final_answer()` 会调用 `tools.pddl_validator.clean_plan_output()`，从模型回复中抽取完整的括号动作行。原始模型回复仍保留在 `Trajectory.messages` 中，便于后续人工检查或接入真正的 planner validator。

当前 PDDL metrics 是轻量格式验证，不调用外部 planner：

- `non_empty_plan`
- `valid_action_format`
- `no_variables`
- `uses_known_actions`
- `uses_declared_objects`
- `plan_length`
- `exact_plan_match`，仅当有 reference plan 时计算。

## 10. 真实 ALFWorld / ScienceWorld Adapter

`AlfworldEnvironment`：

- 依赖真实 `alfworld` 包。
- 使用 `alfworld.agents.environment.get_environment(env_type)`。
- 按 ALFWorld batch API 调用 `init_env(batch_size=1)`。
- `reset()` 读取 batch observation 和 `admissible_commands`。
- `step(action)` 会把单个 action 包成 list 传给 ALFWorld。
- split 会映射到 ALFWorld 官方的 `train_eval` 名称：`valid_seen/dev/validation -> eval_in_distribution`，`valid_unseen/test -> eval_out_of_distribution`。
- 默认真实配置是 `configs/environments/alfworld_textworld_base.yaml`。
- 需要设置 `ALFWORLD_DATA`，它必须指向包含 `json_2.1.1/` 和 `logic/` 的目录。

配置示例：

```yaml
environment:
  provider: alfworld
  config:
    config_path: configs/environments/alfworld_textworld_base.yaml
    env_type: AlfredTWEnv
    batch_size: 1
```

`ScienceWorldEnvironment`：

- 依赖真实 `scienceworld` 包。
- 按官方示例使用 `ScienceWorldEnv("", jar_path, envStepLimit=...)`。
- 使用 `ScienceWorldEnv.load(task_name_or_id, variation_id, simplification, generateGoldPath)`。
- `reset()` 返回初始 observation。
- `step(action)` 返回 observation、reward、done、info。
- 会尝试从 `info["valid"]`、`get_valid_action_object_combinations()` 或 `get_valid_action_object_combinations_with_templates()` 获取 admissible actions。
- 真实小样本 manifest 在 `data/real_samples/scienceworld_manifest_small.json`。

配置示例：

```yaml
environment:
  provider: scienceworld
  config:
    env_step_limit: 50
    simplification: easy
    generate_gold_path: false
```

这些 adapter 的测试使用 fake module，不需要安装真实 ALFWorld / ScienceWorld。

## 11. LLM Provider

`OpenAICompatibleLLM` 自动读取 `.env`，字段包括：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`

config 支持：

```yaml
llm:
  provider: openai-compatible
  model: ${OPENAI_MODEL}
  timeout: 30
  retries: 0
  temperature: 0
  max_tokens: 128
  extra_body:
    enable_thinking: false
```

`extra_body` 会合并进 `/chat/completions` payload。它用于兼容 Qwen/SiliconFlow 等 provider 的非标准参数，例如关闭 thinking。

## 12. Memory 边界

当前只使用：

```python
NullMemoryProvider
```

行为：

- `retrieve()` 返回空列表。
- `update()` 什么都不做。

memory provider 现在通过 registry 创建：

```python
@registry.register_memory_provider("null")
class NullMemoryProvider(MemoryProvider):
    ...
```

runner 使用 `registry.get_memory_provider(config.memory.provider)` 创建 provider。当前只注册 `"null"`，因此行为仍然是 no-memory。后续接入真实 memory 方法时，只需要新增 provider 类并注册，不应该改 dataset builder、task 或 MAS 主流程。当前不得实现真实 memory 逻辑。

## 13. Runner 与 Artifacts

每次运行输出：

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

模型名会经过 `safe_slug()` 清洗，所以 `Qwen/Qwen3-8B` 会变成 `Qwen_Qwen3-8B`。

## 14. 常用命令

PDDL 真实小样本：

```powershell
python -m mas_scope.cli validate-data --builder pddl --data-path data/real_samples/pddl/ipc-2000/blocks-strips-typed --split default --limit 2
python -m mas_scope.cli run --config configs/experiments/pddl_real2_autogen_mock.yaml
python -m mas_scope.cli run --config configs/experiments/pddl_real2_autogen_openai.yaml
```

QA 真实小样本：

```powershell
python -m mas_scope.cli run --config configs/experiments/hotpotqa_real5_autogen_openai.yaml
python -m mas_scope.cli run --config configs/experiments/strategyqa_real5_autogen_openai.yaml
```

环境 adapter 检查：

```powershell
python -m mas_scope.cli validate-env --environment alfworld-mock
python -m mas_scope.cli validate-env --environment scienceworld-mock
```

真实 ALFWorld / ScienceWorld 小样本配置：

```powershell
python -m mas_scope.cli run --config configs/experiments/alfworld_valid_seen_real3_autogen_openai.yaml
python -m mas_scope.cli run --config configs/experiments/alfworld_valid_unseen_real3_autogen_openai.yaml
python -m mas_scope.cli run --config configs/experiments/scienceworld_dev_real3_autogen_openai.yaml
```

当前本机如果没有安装真实包或没有设置数据路径，会直接报清晰错误，不会退回 mock。

## 15. 代码重构说明

已经清理的冗余：

- 删除未使用的通用 `Registry` 类。
- 删除未使用的 `build_agent_prompt()`。
- 删除未使用的 `prompts/default_roles.py`。
- 统一新代码从 `core.types` 导入共享 schema。
- 增加 `.gitignore`，忽略 `.env`、`runs/`、`__pycache__/`、`.pytest_cache/`。

保留的兼容代码：

- legacy dataset adapters：`HotpotQAAdapter`、`StrategyQAAdapter`、`PDDLAdapter`。
- schema re-export 文件：`datasets/schema.py`、`mas/messages.py`、`environments/schema.py`。

## 16. 后续优先事项

建议下一步：

1. 增加 `inspect-run` CLI。
2. 补 `macnet/camel/dylan` 的真实小样本配置，对比 backbone。
3. 在 aggregate metrics 中加入 `num_errors`、`error_rate`、`mean_total_tokens`、`mean_completion_tokens`。
4. 用本地已安装的真实 ALFWorld / ScienceWorld 包做小规模 smoke run。
5. 后续再做 benchmark-scale 实验。

## 17. 当前测试

运行：

```powershell
D:\miniconda\python.exe -m pytest
```

当前通过：

```text
36 passed
```
