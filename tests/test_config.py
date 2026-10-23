"""Tests for where settings come from (ai_playground/config.py)."""

from ai_playground.config import Settings


def test_defaults_without_env_file(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.llm_base_url == "http://localhost:1234/v1"
    assert settings.agent_max_turns == 10


def test_environment_variable_overrides_default(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "qwen/qwen3.6-35b-a3b")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.llm_model == "qwen/qwen3.6-35b-a3b"


def test_env_file_is_read_and_unrelated_lines_are_ignored(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_MAX_TURNS", raising=False)
    env_file = tmp_path / ".env"
    # NEXTAUTH_SECRET is for the Langfuse container, not for us.
    env_file.write_text("AGENT_MAX_TURNS=4\nNEXTAUTH_SECRET=abc\n")

    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]

    assert settings.agent_max_turns == 4


def test_values_passed_in_code_win(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "from-environment")

    settings = Settings(_env_file=None, llm_model="from-code")  # type: ignore[call-arg]

    assert settings.llm_model == "from-code"
