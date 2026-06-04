"""No-memory baseline experiment runner."""

from __future__ import annotations

from pathlib import Path

from mas_scope.config.loader import load_experiment_config
from mas_scope.config.schema import ExperimentConfig
from mas_scope.core.ids import get_git_commit, safe_slug, timestamp_slug
from mas_scope.core.registry import registry
from mas_scope.core.types import TaskResult
from mas_scope.environments.alfworld import AlfworldEnvironment
from mas_scope.environments.mock_alfworld import MockAlfworldEnvironment
from mas_scope.environments.mock_scienceworld import MockScienceWorldEnvironment
from mas_scope.environments.scienceworld import ScienceWorldEnvironment
from mas_scope.evaluation.aggregate import aggregate_metrics
from mas_scope.execution.artifacts import ArtifactWriter
from mas_scope.llm.mock import MockLLM
from mas_scope.llm.openai_compatible import OpenAICompatibleLLM
from mas_scope.tasks.formal_planning import FormalPlanningTask
from mas_scope.tasks.interactive import InteractiveTask
from mas_scope.tasks.qa import QATask


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig | str | Path) -> None:
        self.config = load_experiment_config(config) if not isinstance(config, ExperimentConfig) else config

    def run(self) -> Path:
        config = self.config
        builder_cls = registry.get_dataset_builder(config.dataset.builder)
        builder = builder_cls(config.dataset.data_path)
        build_limit = None if config.dataset.limit is None else config.dataset.limit + config.dataset.offset
        examples = builder.build(config.dataset.split, build_limit)
        if config.dataset.offset:
            examples = examples[config.dataset.offset :]
        environment_name = self.config.environment.provider or (examples[0].metadata.get("environment") if examples else None)
        task = self._create_task(config.task.type, environment_name)
        llm = self._create_llm()
        mas = registry.get_mas(config.mas.type)(mas_config=config.mas.config)
        memory = self._create_memory()
        memory_provider_name = config.memory.provider or "null"
        run_dir = self._run_dir()
        writer = ArtifactWriter(run_dir)
        writer.write_yaml("config.yaml", config.model_dump(mode="json"))
        writer.write_json(
            "manifest.json",
            {
                "timestamp": timestamp_slug(),
                "experiment_name": config.experiment_name,
                "dataset_builder": config.dataset.builder,
                "split": config.dataset.split,
                "mas_type": config.mas.type,
                "model": llm.model_name,
                "memory_provider": memory_provider_name,
                "memory_config": config.memory.config,
                "num_examples": len(examples),
                "git_commit": get_git_commit(Path.cwd()),
            },
        )

        results: list[TaskResult] = []
        for example in examples:
            writer.append_jsonl("examples.jsonl", example)
            try:
                if example.task_type == "interactive":
                    trajectory = task.run_episode(example, mas, llm, memory)
                    prediction = task.parse_final_answer(trajectory)
                    metrics = task.evaluate_trajectory(trajectory)
                else:
                    trajectory = mas.run(example, task, llm, memory)
                    prediction = task.parse_final_answer(trajectory)
                    metrics = task.evaluate(prediction, example)
                result = TaskResult(
                    run_id=trajectory.run_id,
                    example_id=example.example_id,
                    prediction=prediction,
                    target=example.target,
                    metrics=metrics,
                    success=self._success(metrics),
                    cost=trajectory.usage,
                )
                memory.update(example, trajectory, result)
                writer.append_jsonl("trajectories.jsonl", trajectory)
                for step in trajectory.environment_steps:
                    writer.append_jsonl("environment_steps.jsonl", step)
            except Exception as exc:
                writer.append_jsonl("errors.jsonl", {"example_id": example.example_id, "error": str(exc)})
                result = TaskResult(
                    run_id="",
                    example_id=example.example_id,
                    prediction=None,
                    target=example.target,
                    metrics={},
                    success=False,
                    error=str(exc),
                )
            writer.append_jsonl("results.jsonl", result)
            results.append(result)
        writer.write_json("metrics.json", aggregate_metrics(results))
        return run_dir

    def _create_task(self, task_type: str, environment_name: str | None):
        if task_type == "qa":
            return QATask()
        if task_type == "formal_planning":
            return FormalPlanningTask()
        return InteractiveTask(self._create_environment(environment_name), max_steps=self.config.task.max_steps)

    def _create_environment(self, environment_name: str | None):
        env_config = dict(self.config.environment.config)
        if environment_name in (None, "alfworld-mock"):
            return MockAlfworldEnvironment(max_steps=self.config.task.max_steps)
        if environment_name == "scienceworld-mock":
            return MockScienceWorldEnvironment(max_steps=self.config.task.max_steps)
        if environment_name == "alfworld":
            return AlfworldEnvironment(**env_config)
        if environment_name == "scienceworld":
            env_config.setdefault("env_step_limit", self.config.task.max_steps)
            return ScienceWorldEnvironment(**env_config)
        raise KeyError(f"Unknown environment '{environment_name}'")

    def _create_llm(self):
        if self.config.llm.provider == "mock":
            return MockLLM(model_name=self.config.llm.model)
        return OpenAICompatibleLLM(
            model_name=self.config.llm.model,
            timeout=self.config.llm.timeout,
            retries=self.config.llm.retries,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            extra_body=self.config.llm.extra_body,
        )

    def _create_memory(self):
        provider_cls = registry.get_memory_provider(self.config.memory.provider)
        return provider_cls(**self.config.memory.config)

    def _run_dir(self) -> Path:
        name = "_".join(
            [
                timestamp_slug(),
                safe_slug(self.config.dataset.builder),
                safe_slug(self.config.mas.type),
                safe_slug(self.config.llm.model),
            ]
        )
        return self.config.output.dir / name

    def _success(self, metrics: dict) -> bool:
        if "success" in metrics:
            return bool(metrics["success"])
        if metrics.get("scored") is False:
            return True
        if "exact_match" in metrics:
            return bool(metrics["exact_match"])
        if "valid_action_format" in metrics:
            return bool(metrics["valid_action_format"])
        return False
