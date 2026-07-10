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

## 9. 常用命令

安装开发依赖：

```powershell
python -m pip install -e ".[dev]"
```

验证数据集：

```powershell
python -m mas_scope.cli validate-data --builder hotpotqa --data-path data/samples/hotpotqa_sample.json --split dev --limit 2
python -m mas_scope.cli validate-data --builder strategyqa --data-path data/samples/strategyqa_sample.json --split dev --limit 2
python -m mas_scope.cli validate-data --builder pddl --data-path data/samples/pddl_sample.jsonl --split test --limit 1
```

运行一个无记忆 smoke baseline：

```powershell
python -m mas_scope.cli run --config configs/experiments/smoke_hotpotqa_autogen.yaml
```

从 run artifacts 构造旧式逐消息 memory bank：

```powershell
python -m mas_scope.cli build-memory-bank --run-dir runs/example_run --output runs/memory_banks/seed.jsonl
```

这条路径主要用于调试或兼容旧实验；主线建议使用 ReasoningBank-style memory。

## 10. ReasoningBank-style Memory 抽取

主抽取入口是根目录的 `extract_memories.py`。输入应优先使用一次 MAS run 目录：

```text
runs/example_run/
  examples.jsonl
  trajectories.jsonl
  results.jsonl
```

运行命令：

```powershell
python extract_memories.py `
  --input_dir runs/example_run `
  --output_file data/memories_hotpotqa_reasoningbank.jsonl `
  --model Qwen/Qwen3-8B `
  --overwrite
```

抽取结果是 JSONL，每行一条 memory：

```json
{
  "memory_id": "m_000001",
  "source": {
    "trajectory_id": "trajectory_id",
    "example_id": "hotpotqa_example_id",
    "dataset_name": "hotpotqa",
    "mas_type": "macnet",
    "producer_agent": "multi-agent",
    "producer_role": "team",
    "success": false
  },
  "source_task_description": "source task or local situation",
  "condition": "when this memory applies",
  "experience": "reusable lesson, warning, or strategy",
  "evidence": "trajectory behavior or result supporting the memory"
}
```

抽取阶段只负责提炼经验，不决定未来分配给哪个 agent。因此 memory 中不应包含：

- `agent_mask`
- `target_agent`
- `scope`
- `allocation_score`

这些字段属于 allocator 或 runtime memory provider 的职责。

## 11. HotpotQA + Multi-agent 适配点

HotpotQA 抽取会额外关注：

- bridge entity 是否停在中间实体。
- comparison 问题是否比较错对象或错属性。
- final answer 是否满足最小 supported span/entity。
- yes/no 问题是否有 evidence support。
- critic 是否真正验证 actor 输出。
- summarizer 是否正确仲裁多个候选答案。

Multi-agent provenance 会保留：

- `agent_name`
- `role`
- `turn_id`
- `agent_order`
- `topology`
- `edges`

团队协作经验使用：

```json
{"producer_agent": "multi-agent", "producer_role": "team"}
```

单个 agent 行为经验使用该 agent 的具体 role，例如：

```json
{"producer_agent": "critic agent 1", "producer_role": "critic agent 1"}
```

## 12. 目录清理约定

源码和测试应提交到 git。以下目录默认是本地运行产物，已经在 `.gitignore` 中忽略：

- `runs/`：实验运行轨迹、结果、metrics。
- `outputs/`：allocator ablation 输出、diagnostics、reports。
- `logs/`：抽取失败响应和调试日志。
- `.tmp/`：临时实验输入输出。
- `.pytest_cache/`、`__pycache__/`：测试和 Python 缓存。

安全清理对象：

- `.pytest_cache/`
- `__pycache__/`
- 空的 `.tmp/`
- 明确无用的 stdout/stderr 临时文件

不建议直接删除：

- `runs/`
- `outputs/`
- `logs/`
- `.env`

这些文件可能用于复现实验、追踪失败样本或保存本地 API 配置。
