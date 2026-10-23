"""All settings in one place.

Each field below can be set by an environment variable with the same name in
capitals (for example LLM_MODEL), or by a line in the .env file in the
directory you run from. Real environment variables win over .env, and both
win over the defaults written here.

Use it like this:

    settings = Settings()                          # read env vars and .env
    settings = Settings(llm_model="qwen/qwen3.6")  # override one value in code
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        # .env also holds settings for the Docker containers; skip those.
        extra="ignore",
    )

    # --- The model server ---------------------------------------------------
    llm_base_url: str = "http://localhost:1234/v1"
    llm_model: str = "google/gemma-4-e4b"
    # Local servers such as LM Studio don't check the key, but the OpenAI
    # client requires one, so any text works there.
    llm_api_key: str = "not-needed-for-local-servers"
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2
    # How much a "thinking" model (such as Qwen 3) reasons before answering.
    # Leave unset for models that don't think: some servers reject it for
    # them. With JSON output, LM Studio files a thinking model's whole answer
    # as its reasoning and sends an empty reply, so set "none" there.
    llm_reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] | None = None

    # --- The agent loop -------------------------------------------------------
    # Stop with an error if the model is still asking for tools after this
    # many turns, instead of looping forever.
    agent_max_turns: int = 10

    # --- Tracing (see tracing.py) ---------------------------------------------
    trace_project_name: str = "agentic-ai-course"
    # The user each trace is recorded for. There are no real users yet, so
    # every run uses this made-up one; a chat app would use each request's
    # real user instead.
    trace_user_id: str = "playground-user"
    phoenix_collector_endpoint: str = "http://localhost:6006"
    langfuse_host: str = "http://localhost:3000"
    # Without these, traces go to Phoenix only.
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
