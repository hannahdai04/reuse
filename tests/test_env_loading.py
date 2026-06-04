import os

from mas_scope.core.env import load_environment
from mas_scope.llm.openai_compatible import OpenAICompatibleLLM


def test_load_environment_reads_env_example(tmp_path, monkeypatch):
    env_file = tmp_path / ".env.example"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=test-key",
                "OPENAI_BASE_URL=https://example.test/v1",
                "OPENAI_MODEL=test-model",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    loaded = load_environment()

    assert loaded["OPENAI_API_KEY"] == "test-key"
    assert os.environ["OPENAI_BASE_URL"] == "https://example.test/v1"
    assert os.environ["OPENAI_MODEL"] == "test-model"


def test_openai_compatible_llm_uses_loaded_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env.example"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=file-key",
                "OPENAI_BASE_URL=https://example.test/v1",
                "OPENAI_MODEL=file-model",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    llm = OpenAICompatibleLLM()

    assert llm.api_key == "file-key"
    assert llm.base_url == "https://example.test/v1"
    assert llm.model_name == "file-model"


def test_openai_compatible_llm_stores_generation_defaults(tmp_path, monkeypatch):
    env_file = tmp_path / ".env.example"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=file-key",
                "OPENAI_BASE_URL=https://example.test/v1",
                "OPENAI_MODEL=file-model",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    llm = OpenAICompatibleLLM(temperature=0.2, max_tokens=32, extra_body={"enable_thinking": False})

    assert llm.temperature == 0.2
    assert llm.max_tokens == 32
    assert llm.extra_body == {"enable_thinking": False}
