"""Dataset adapter factory."""

from __future__ import annotations

from pathlib import Path

from mas_scope.datasets.base import BaseDatasetAdapter
from mas_scope.datasets.hotpotqa import HotpotQAAdapter
from mas_scope.datasets.pddl import PDDLAdapter
from mas_scope.datasets.strategyqa import StrategyQAAdapter


DATASET_ADAPTERS: dict[str, type[BaseDatasetAdapter]] = {
    "hotpotqa": HotpotQAAdapter,
    "strategyqa": StrategyQAAdapter,
    "pddl": PDDLAdapter,
}


def create_dataset_adapter(name: str, data_path: str | Path) -> BaseDatasetAdapter:
    try:
        adapter_cls = DATASET_ADAPTERS[name]
    except KeyError as exc:
        known = ", ".join(sorted(DATASET_ADAPTERS))
        raise KeyError(f"Unknown dataset '{name}'. Known datasets: {known}") from exc
    return adapter_cls(data_path)
