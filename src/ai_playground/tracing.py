"""Everything about tracing lives in this file.

A trace is the record of one run of the agent. It is made of spans: one span
per step, each with a start time, an end time, and details we attach. Spans
started inside another span become its children, so Phoenix shows the run as
a tree:

  agent                                  <- one whole call_llm() run
  ├── turn 1                             <- one trip to the model and back
  │   ├── Response                       <- the model call (added automatically)
  │   ├── get_current_time               <- a tool the model asked for
  │   └── get_current_time_in_timezone
  └── turn 2
      └── Response                       <- the model gives its final answer

This file has two parts:
  1. setup_tracing() decides where traces are sent: Phoenix, and Langfuse
     too when its keys are set, so you can compare the same run in both.
     To add or switch tools (Jaeger, ...), this is the part to change.
  2. span() and the helpers built on it (agent_span, turn_span, tool_span)
     create our own spans, so the rest of the code never needs to know
     attribute names.

See docs/tracing.md for the full explanation.
"""

import base64
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum

from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.trace import Span

from ai_playground.config import Settings

# ---------------------------------------------------------------------------
# 1. Setup
#
# OpenTelemetry pieces, from the inside out:
#   - an exporter sends finished spans to one tool, over HTTP (OTLP is the
#     standard format every tool accepts);
#   - a span processor collects spans and hands them to its exporter in
#     batches, so tracing doesn't slow the program down;
#   - the tracer provider owns the processors. Every span goes to every
#     processor, so with two exporters each span is sent to both tools.
# ---------------------------------------------------------------------------


def phoenix_exporter(settings: Settings) -> SpanExporter:
    """Exporter for a Phoenix server; by default the local one on port 6006."""
    return OTLPSpanExporter(endpoint=f"{settings.phoenix_collector_endpoint}/v1/traces")


def langfuse_auth_header(public_key: str, secret_key: str) -> str:
    """Langfuse checks the keys with "Basic" authentication: the two keys
    joined by a colon, then base64-encoded."""
    credentials = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    return f"Basic {credentials}"


def langfuse_exporter(settings: Settings, public_key: str, secret_key: str) -> SpanExporter:
    """Exporter for Langfuse; by default the local one on port 3000."""
    return OTLPSpanExporter(
        endpoint=f"{settings.langfuse_host}/api/public/otel/v1/traces",
        headers={"Authorization": langfuse_auth_header(public_key, secret_key)},
    )


def make_exporters(settings: Settings) -> list[SpanExporter]:
    """One exporter per tool that traces should go to."""
    exporters = [phoenix_exporter(settings)]
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        exporters.append(
            langfuse_exporter(settings, settings.langfuse_public_key, settings.langfuse_secret_key)
        )
    return exporters


# The provider created by setup_tracing(), so calling it again does nothing.
_tracer_provider: TracerProvider | None = None


def setup_tracing(settings: Settings | None = None) -> TracerProvider:
    """Send traces to Phoenix, and also to Langfuse if its keys are set.

    Call this once, at the start of the program, before calling the model.
    Calling it again (for example, re-running a notebook cell) is harmless.
    """
    global _tracer_provider
    if _tracer_provider is not None:
        return _tracer_provider
    if settings is None:
        settings = Settings()

    # The resource describes who is sending the spans. Phoenix uses this
    # attribute to group traces into a project.
    resource = Resource.create({"openinference.project.name": settings.trace_project_name})
    tracer_provider = TracerProvider(resource=resource)
    for exporter in make_exporters(settings):
        tracer_provider.add_span_processor(BatchSpanProcessor(exporter))

    # Make this the provider that trace.get_tracer() (below) uses.
    trace.set_tracer_provider(tracer_provider)

    # Automatically create a span for every call made with the OpenAI client.
    # These spans hold the prompt, the model's reply, the tool definitions and
    # the token counts, without us writing any tracing code for them.
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

    _tracer_provider = tracer_provider
    return tracer_provider


# ---------------------------------------------------------------------------
# 2. Our own spans
#
# Attribute names come from OpenInference, the naming scheme Phoenix uses and
# Langfuse understands. Because they are only written here, changing scheme
# (for example to OpenTelemetry's "gen_ai.*" names) means changing only this
# file.
#
# If setup_tracing() was never called, these spans do nothing, so the agent
# code also works without Phoenix (for example in the tests).
# ---------------------------------------------------------------------------

SPAN_KIND = "openinference.span.kind"
INPUT = "input.value"
INPUT_MIME_TYPE = "input.mime_type"
OUTPUT = "output.value"
TOOL_NAME = "tool.name"


class SpanKind(StrEnum):
    """The kinds of step OpenInference defines. Tools use them for labels and icons."""

    AGENT = "AGENT"  # a whole agent run, from question to answer
    CHAIN = "CHAIN"  # a group of related steps, such as one turn
    TOOL = "TOOL"  # one tool call
    LLM = "LLM"  # one model call (the OpenAI instrumentor creates these)
    RETRIEVER = "RETRIEVER"  # fetching documents, for example in RAG
    EMBEDDING = "EMBEDDING"  # turning text into vectors
    RERANKER = "RERANKER"  # re-ordering retrieved documents by relevance
    GUARDRAIL = "GUARDRAIL"  # checking input or output against rules
    EVALUATOR = "EVALUATOR"  # scoring an answer


tracer = trace.get_tracer(__name__)


# @contextmanager turns a function into something you can use with "with".
# The code before "yield" runs when the "with" block starts, and the span
# ends when the "with" block ends.


@contextmanager
def span(name: str, kind: SpanKind, input_value: str | None = None) -> Iterator[Span]:
    """Start a span of any kind. The helpers below are shortcuts for common ones."""
    with tracer.start_as_current_span(name) as current:
        current.set_attribute(SPAN_KIND, kind)
        if input_value is not None:
            current.set_attribute(INPUT, input_value)
        yield current


@contextmanager
def agent_span(prompt: str) -> Iterator[Span]:
    """Span for one whole agent run, from the user's prompt to the final answer."""
    with span("agent", SpanKind.AGENT, prompt) as current:
        yield current


@contextmanager
def turn_span(number: int) -> Iterator[Span]:
    """Span for one trip to the model and back, including the tools it asked for."""
    with span(f"turn {number}", SpanKind.CHAIN) as current:
        yield current


@contextmanager
def tool_span(name: str, arguments: str) -> Iterator[Span]:
    """Span for one tool call. The arguments are the JSON text the model sent."""
    with span(name, SpanKind.TOOL, arguments) as current:
        current.set_attribute(TOOL_NAME, name)
        current.set_attribute(INPUT_MIME_TYPE, "application/json")
        yield current


def set_output(current: Span, value: str) -> None:
    """Record what a step produced, so the tools can show it next to the input."""
    current.set_attribute(OUTPUT, value)
