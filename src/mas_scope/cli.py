"""Command line interface for no-memory MAS baselines."""

from __future__ import annotations

from pathlib import Path

import typer

from mas_scope.core.env import load_environment
from mas_scope.core.registry import registry
from mas_scope.environments.alfworld import AlfworldEnvironment
from mas_scope.environments.mock_alfworld import MockAlfworldEnvironment
from mas_scope.environments.mock_scienceworld import MockScienceWorldEnvironment
from mas_scope.environments.scienceworld import ScienceWorldEnvironment
from mas_scope.execution.runner import ExperimentRunner
from mas_scope.memory.bank import build_memory_bank

load_environment()

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command("validate-data")
def validate_data(
    builder: str = typer.Option(..., "--builder"),
    data_path: Path = typer.Option(..., "--data-path"),
    split: str = typer.Option("dev", "--split"),
    limit: int | None = typer.Option(None, "--limit"),
) -> None:
    try:
        builder_cls = registry.get_dataset_builder(builder)
        examples = builder_cls(data_path).build(split, limit)
    except Exception as exc:
        _fail(exc)
    typer.echo(f"validated {len(examples)} examples for builder={builder} split={split}")


@app.command("list-splits")
def list_splits(
    builder: str = typer.Option(..., "--builder"),
    data_path: Path = typer.Option(..., "--data-path"),
) -> None:
    try:
        builder_cls = registry.get_dataset_builder(builder)
        splits = builder_cls(data_path).available_splits()
    except Exception as exc:
        _fail(exc)
    for split in splits:
        typer.echo(split)


@app.command("validate-env")
def validate_env(
    environment: str = typer.Option(..., "--environment"),
    builder: str | None = typer.Option(None, "--builder"),
    data_path: Path | None = typer.Option(None, "--data-path"),
    split: str = typer.Option("dev", "--split"),
    env_config_path: Path | None = typer.Option(None, "--env-config-path"),
    max_steps: int = typer.Option(30, "--max-steps"),
) -> None:
    example = None
    try:
        if builder and data_path:
            builder_cls = registry.get_dataset_builder(builder)
            examples = builder_cls(data_path).build(split, limit=1)
            example = examples[0] if examples else None
        env = _create_environment(environment, env_config_path, max_steps)
        if example is not None:
            state = env.reset(example)
            typer.echo(f"validated environment={environment} observation={state.observation[:120]!r}")
        else:
            typer.echo(f"validated environment={environment}")
        env.close()
    except Exception as exc:
        _fail(exc)


@app.command("run")
def run(config: Path = typer.Option(..., "--config")) -> None:
    try:
        run_dir = ExperimentRunner(config).run()
    except Exception as exc:
        _fail(exc)
    typer.echo(str(run_dir))


@app.command("build-memory-bank")
def build_memory_bank_command(
    run_dir: Path = typer.Option(..., "--run-dir"),
    output: Path = typer.Option(..., "--output"),
    append: bool = typer.Option(False, "--append"),
) -> None:
    try:
        count = build_memory_bank(run_dir, output, append=append)
    except Exception as exc:
        _fail(exc)
    typer.echo(f"wrote {count} memory units to {output}")


def _create_environment(environment: str, env_config_path: Path | None, max_steps: int):
    if environment == "alfworld-mock":
        return MockAlfworldEnvironment(max_steps=max_steps)
    if environment == "scienceworld-mock":
        return MockScienceWorldEnvironment(max_steps=max_steps)
    if environment == "alfworld":
        return AlfworldEnvironment(config_path=env_config_path, batch_size=1)
    if environment == "scienceworld":
        return ScienceWorldEnvironment(env_step_limit=max_steps)
    raise KeyError(f"Unknown environment '{environment}'")


def _fail(exc: Exception) -> None:
    typer.secho(f"error: {exc}", err=True, fg=typer.colors.RED)
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
