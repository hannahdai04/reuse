import pytest

import mas_scope.mas  # noqa: F401 - imports register MAS classes
from mas_scope.core.registry import registry
from mas_scope.core.types import ActionDecision, TaskExample, Trajectory
from mas_scope.llm.mock import MockLLM
from mas_scope.mas.agent import MASAgent
from mas_scope.mas.autogen_style import AutoGenStyleMAS
from mas_scope.mas.camel_style import CAMELStyleMAS
from mas_scope.mas.dylan_style import DyLANStyleMAS
from mas_scope.mas.macnet_style import MacNetStyleMAS
from mas_scope.memory.null_memory import NullMemoryProvider
from mas_scope.tasks.qa import QATask


def qa_example():
    return TaskExample(
        example_id="qa_001",
        dataset_name="hotpotqa",
        task_type="qa",
        split="dev",
        input={"question": "What is the capital of France?", "context": "Paris is the capital of France."},
        target={"answer": "Paris"},
    )


def deps():
    return qa_example(), QATask(), MockLLM(), NullMemoryProvider()


def test_registry_returns_registered_mas_classes():
    assert registry.get_mas("autogen") is AutoGenStyleMAS
    assert registry.get_mas("macnet") is MacNetStyleMAS
    assert registry.get_mas("camel") is CAMELStyleMAS
    assert registry.get_mas("dylan") is DyLANStyleMAS
    assert set(registry.names()) >= {"autogen", "macnet", "camel", "dylan"}


def test_mas_agent_missing_template_field_has_clear_error():
    agent = MASAgent(
        role="test agent",
        topology_node_id=0,
        system_prompt_template="System {missing}",
        user_prompt_template="User {value}",
    )

    with pytest.raises(ValueError, match="test agent.*system_prompt_template.*missing"):
        agent.invoke(MockLLM(), system_inputs={}, user_inputs={"value": "ok"}, turn_id=0)


def test_autogen_init_and_run():
    example, task, llm, memory = deps()
    mas = AutoGenStyleMAS()

    assert len(mas.agents_list) == 2
    assert [agent.role for agent in mas.agents_list] == ["assistant agent", "user proxy agent"]

    trajectory = mas.run(example, task, llm, memory)

    assert isinstance(trajectory, Trajectory)
    assert len(trajectory.messages) == 2
    assert trajectory.messages[-1].role == "user proxy agent"
    assert trajectory.final_answer == trajectory.messages[-1].content
    assert trajectory.metadata["mas_style"] == "autogen"


def test_macnet_init_and_run():
    example, task, llm, memory = deps()
    mas = MacNetStyleMAS()

    roles = [agent.role for agent in mas.agents_list]
    assert len(mas.agents_list) == 5
    assert set(roles) == {
        "actor agent 1",
        "actor agent 2",
        "critic agent 1",
        "critic agent 2",
        "summarizer agent",
    }

    trajectory = mas.run(example, task, llm, memory)

    assert isinstance(trajectory, Trajectory)
    assert len(trajectory.messages) == 5
    assert trajectory.messages[-1].role == "summarizer agent"
    assert trajectory.final_answer == trajectory.messages[-1].content
    assert trajectory.metadata["mas_style"] == "macnet"
    assert trajectory.metadata["edges"]


def test_camel_run_returns_trajectory():
    example, task, llm, memory = deps()
    trajectory = CAMELStyleMAS().run(example, task, llm, memory)

    assert isinstance(trajectory, Trajectory)
    assert trajectory.metadata["mas_style"] == "camel"
    assert trajectory.messages


def test_dylan_run_returns_dynamic_trace():
    example, task, llm, memory = deps()
    trajectory = DyLANStyleMAS().run(example, task, llm, memory)

    assert isinstance(trajectory, Trajectory)
    assert trajectory.metadata["mas_style"] == "dylan"
    assert trajectory.metadata["selected_agents"]
    assert trajectory.metadata["selection_trace"]


@pytest.mark.parametrize(
    ("mas_cls", "style"),
    [
        (AutoGenStyleMAS, "autogen"),
        (MacNetStyleMAS, "macnet"),
        (CAMELStyleMAS, "camel"),
        (DyLANStyleMAS, "dylan"),
    ],
)
def test_act_returns_action_decision(mas_cls, style):
    example, task, llm, memory = deps()
    decision = mas_cls().act(
        example=example,
        observation="You are in a kitchen. You see an apple on the table.",
        task=task,
        llm=llm,
        memory=memory,
        context={},
    )

    assert isinstance(decision, ActionDecision)
    assert decision.action
    assert decision.messages
    assert decision.metadata["mas_style"] == style
