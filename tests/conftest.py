"""Shared test setup. pytest loads this file automatically.

Fixtures defined here can be used by any test just by naming them as an
argument, for example `def test_something(settings, model): ...`.
"""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ai_playground.config import Settings
from tests.fakes import FakeModel

# Record spans in memory instead of sending them anywhere, so tests can check
# them. OpenTelemetry allows setting the provider only once per process, so
# this runs once, when pytest loads this file.
_span_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_span_exporter))
trace.set_tracer_provider(_provider)


@pytest.fixture
def span_exporter() -> InMemorySpanExporter:
    """Holds the spans finished during the test: call .get_finished_spans()."""
    _span_exporter.clear()
    return _span_exporter


@pytest.fixture
def settings() -> Settings:
    """Settings for tests, ignoring .env so results don't depend on your machine."""
    return Settings(
        _env_file=None,  # type: ignore[call-arg]  # a pydantic-settings option, not a field
        llm_base_url="http://example.test/v1",
        llm_model="test-model",
        llm_api_key="test-key",
        llm_timeout_seconds=5,
        llm_max_retries=0,
        agent_max_turns=3,
        trace_user_id="test-user",
        langfuse_public_key=None,
        langfuse_secret_key=None,
    )


@pytest.fixture
def model() -> Iterator[FakeModel]:
    """A fake model server. Queue replies in `model.replies`; read `model.requests`."""
    fake = FakeModel()
    client = MagicMock()
    client.responses.create.side_effect = fake.create
    client.__enter__.return_value = client
    with patch("ai_playground.llm.OpenAI", return_value=client) as constructor:
        fake.client = client
        fake.constructor = constructor
        yield fake
