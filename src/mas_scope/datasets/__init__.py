"""Dataset adapters."""

from mas_scope.core.types import TaskExample
from mas_scope.datasets.hotpotqa import HotpotQAAdapter
from mas_scope.datasets.pddl import PDDLAdapter
from mas_scope.datasets.strategyqa import StrategyQAAdapter

__all__ = ["TaskExample", "HotpotQAAdapter", "StrategyQAAdapter", "PDDLAdapter"]
