import sys
import types

from mas_scope.core.types import TaskExample
from mas_scope.environments.alfworld import AlfworldEnvironment
from mas_scope.environments.scienceworld import ScienceWorldEnvironment


def _example(dataset_name, payload, metadata=None):
    return TaskExample(
        example_id="episode_1",
        dataset_name=dataset_name,
        task_type="interactive",
        split="valid_seen",
        input=payload,
        target={"success": True},
        metadata=metadata or {},
    )


def test_alfworld_environment_adapter_with_fake_module(monkeypatch):
    class FakeBatchEnv:
        def reset(self):
            return ["obs"], {"admissible_commands": [["look", "go north"]], "score": [0]}

        def step(self, actions):
            assert actions == ["look"]
            return ["done obs"], [1], [True], {"admissible_commands": [["look"]], "won": [True]}

        def close(self):
            self.closed = True

    class FakeEnvFactory:
        def __init__(self, config, train_eval):
            self.config = config
            self.train_eval = train_eval
            assert train_eval == "eval_in_distribution"

        def init_env(self, batch_size):
            assert batch_size == 1
            return FakeBatchEnv()

    def get_environment(env_type):
        assert env_type == "AlfredTWEnv"
        return FakeEnvFactory

    alfworld = types.ModuleType("alfworld")
    agents = types.ModuleType("alfworld.agents")
    environment = types.ModuleType("alfworld.agents.environment")
    environment.get_environment = get_environment
    modules = types.ModuleType("alfworld.agents.modules")
    generic = types.ModuleType("alfworld.agents.modules.generic")
    monkeypatch.setitem(sys.modules, "alfworld", alfworld)
    monkeypatch.setitem(sys.modules, "alfworld.agents", agents)
    monkeypatch.setitem(sys.modules, "alfworld.agents.environment", environment)
    monkeypatch.setitem(sys.modules, "alfworld.agents.modules", modules)
    monkeypatch.setitem(sys.modules, "alfworld.agents.modules.generic", generic)

    env = AlfworldEnvironment(config={"env": {"type": "AlfredTWEnv"}})
    state = env.reset(_example("alfworld", {"instruction": "look"}))
    assert state.observation == "obs"
    assert state.admissible_actions == ["look", "go north"]

    next_state = env.step("look")
    assert next_state.done is True
    assert next_state.metadata["success"] is True


def test_scienceworld_environment_adapter_with_fake_module(monkeypatch):
    class FakeScienceWorldEnv:
        def __init__(self, env_name, jar_path, **kwargs):
            assert env_name == ""
            assert jar_path is None
            assert kwargs["envStepLimit"] == 3
            self.loaded = None

        def load(self, task_name, variation_id, simplification, generate_gold_path):
            self.loaded = (task_name, variation_id, simplification, generate_gold_path)

        def reset(self):
            return "lab obs", {"score": 0, "valid": ["look", "measure"]}

        def step(self, action):
            assert action == "measure"
            return "done obs", 100, True, {"score": 100, "valid": ["look"]}

        def close(self):
            self.closed = True

    module = types.ModuleType("scienceworld")
    module.ScienceWorldEnv = FakeScienceWorldEnv
    monkeypatch.setitem(sys.modules, "scienceworld", module)

    env = ScienceWorldEnvironment(env_step_limit=3)
    state = env.reset(_example("scienceworld", {"task_name": "measure-temp", "variation_id": 2}))
    assert state.observation == "lab obs"
    assert state.admissible_actions == ["look", "measure"]

    next_state = env.step("measure")
    assert next_state.done is True
    assert next_state.metadata["success"] is True
