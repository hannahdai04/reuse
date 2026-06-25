# 代码文档

## 1. 项目定位

本项目用于研究 LLM 多智能体系统中的跨任务经验记忆复用范围。当前代码可以运行无记忆基线、`llm-scope` 记忆路由实验、ReasoningBank 风格经验抽取，以及 allocator ablation。

核心流程：

```text
DatasetBuilder -> TaskExample -> Task + MAS + LLM + MemoryProvider -> Trajectory -> Evaluation -> Run Artifacts
Run Artifacts -> extract_memories.py -> Experience Memory JSONL -> Retrieval/Masking/Selection/Realization
```

## 2. 目录结构

```text
configs/experiments/        实验 YAML 配置
configs/environments/       可选真实环境配置
data/samples/               离线小样本
data/real_samples/          真实文件小样本和 manifest
src/mas_scope/config/       配置 schema 和 YAML loader
src/mas_scope/core/         共享类型、registry、ID、环境变量加载、异常
src/mas_scope/datasets/     split-aware DatasetBuilder
src/mas_scope/environments/ 环境接口、mock 环境、真实 ALFWorld/ScienceWorld adapter
src/mas_scope/evaluation/   QA / planning / interactive / aggregate metrics
src/mas_scope/execution/    runner 和 artifact writer
src/mas_scope/llm/          LLM interface、MockLLM、OpenAI-compatible provider
src/mas_scope/mas/          MASAgent、BaseMAS、四个 MAS backbone
src/mas_scope/memory/       MemoryProvider、NullMemoryProvider、LLM-scope provider
src/mas_scope/prompts/      MAS prompt templates
src/mas_scope/tasks/        QA、formal planning、interactive task
src/mas_scope/tools/        文本归一化、答案解析、action 解析、PDDL 工具
allocator/                  retrieval、masking、selection、realization、diagnostics
scripts/                    allocator ablation、报告、手动 memory debug
tests/                      离线测试
```

## 3. 共享 Schema

共享类型集中在 `mas_scope.core.types`：

- `TaskExample`：标准化后的可运行样本。
- `AgentSpec`：任务侧看到的 agent 描述。
- `AgentMessage`：单个 agent 的输出消息。
- `ActionDecision`：interactive task 的 action 决策。
- `EnvironmentState` / `EnvironmentStep`：环境状态和 step 日志。
- `Trajectory`：完整 MAS 执行轨迹。
- `TaskResult`：单样本预测、target、metrics、success、cost、error。
- `MessageGraph`：轻量消息图。

新代码应直接从 `mas_scope.core.types` 导入共享类型。

## 4. DatasetBuilder

数据集通过 `mas_scope.core.registry.registry` 注册：

```python
@registry.register_dataset_builder("hotpotqa")
class HotpotQABuilder(DatasetBuilder):
    ...
```

当前 builder：

- `HotpotQABuilder`：支持官方 JSON list、JSONL、Hugging Face 风格 `context.title/context.sentences`。
- `StrategyQABuilder`：支持 JSON/JSONL，并把 bool/string answer 归一化为 `yes/no`。
- `PDDLBuilder`：支持 JSON/JSONL 和 IPC-style 目录结构。
- `ALFWorldBuilder`：索引 `json_2.1.1/{split}/.../traj_data.json`。
- `ScienceWorldBuilder`：读取本地 manifest，不在 builder 阶段 import ScienceWorld。

builder 只负责把原始数据变成 `TaskExample`，不调用 LLM、MAS 或真实环境。

## 5. MAS Framework

MAS 通过 registry 注册：

```python
@registry.register_mas("autogen")
class AutoGenStyleMAS(BaseMAS):
    ...
```

当前保留四个 backbone：

- `autogen`
- `macnet`
- `camel`
- `dylan`

`BaseMAS` 保留两个入口：

- `run(example, task, llm, memory) -> Trajectory`
- `act(example, observation, task, llm, memory, context) -> ActionDecision`

## 6. Memory 与 Experience Extraction

`MemoryProvider` 接口：

- `retrieve(example, agent_spec, context) -> list[dict]`
- `update(example, trajectory, result) -> None`

当前 provider：

- `NullMemoryProvider`：无记忆基线。
- `LLMScopeMemoryProvider`：从 JSONL memory bank 召回候选，并按策略路由给具体 agent。

`extract_memories.py` 从 run artifacts 中抽取 ReasoningBank 风格 experience memory。抽取结果保留 source provenance，但不写入 `agent_mask`、`target_agent`、`scope` 等分配字段；分配逻辑由 memory provider 或 allocator 负责。

## 7. Allocator Ablation

`allocator/` 中的四阶段流程：

```text
Retrieval -> Masking -> Selection -> Realization
```

相关脚本：

- `scripts/run_allocator_ablation.py`
- `scripts/analyze_allocator_ablation.py`
- `scripts/generate_allocator_report.py`
- `scripts/run_manual_memory_debug.py`

这些脚本生成的 records、diagnostics、reports 默认写入 `outputs/`，不应作为源码提交。

## 8. 测试

离线测试不依赖外部 API：

```powershell
python -m pytest
```

清理旧适配器层后，数据集构造的唯一主路径是 `DatasetBuilder`，共享类型的唯一主路径是 `mas_scope.core.types`。
