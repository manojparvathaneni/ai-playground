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

This file has three parts:
  1. setup_tracing() decides where traces are sent: Phoenix, and Langfuse
     too when its keys are set, so you can compare the same run in both.
     To add or switch tools (Jaeger, ...), this is the part to change.
  2. span() and the helpers built on it (agent_span, turn_span, tool_span)
     create our own spans, so the rest of the code never needs to know
     attribute names.
  3. traced_run() groups the traces of one run of a program into a session,
     and labels them with the user, the model and the git commit.
  4. experiment_context() and experiment_item() mark an evaluation's traces
     as an experiment on a Langfuse dataset.

See docs/tracing.md for the full explanation.
"""

import base64
import subprocess
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from openinference.instrumentation import get_attributes_from_context, using_attributes
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import baggage, context, trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Span as SdkSpan
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
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


def langfuse_headers(public_key: str, secret_key: str) -> dict[str, str]:
    """The HTTP headers sent to Langfuse with every batch of spans."""
    return {
        "Authorization": langfuse_auth_header(public_key, secret_key),
        # Tells Langfuse 4 the spans are in its current format, so they can
        # be read back through its API at once (/api/public/v2/observations)
        # instead of after a delay of up to 15 minutes.
        "x-langfuse-ingestion-version": "4",
    }


def langfuse_exporter(settings: Settings, public_key: str, secret_key: str) -> SpanExporter:
    """Exporter for Langfuse; by default the local one on port 3000."""
    return OTLPSpanExporter(
        endpoint=f"{settings.langfuse_host}/api/public/otel/v1/traces",
        headers=langfuse_headers(public_key, secret_key),
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


def setup_tracing(settings: Settings | None = None, *, pydantic_ai: bool = False) -> TracerProvider:
    """Send traces to Phoenix, and also to Langfuse if its keys are set.

    Call this once, at the start of the program, before calling the model.
    Calling it again (for example, re-running a notebook cell) is harmless.

    pydantic_ai=True is for programs that call models through Pydantic AI
    agents (experiment 03); see instrument_pydantic_ai() below.
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
    # Processors run in the order they are added, so these two, which only
    # add labels, come before the ones that send spans away.
    tracer_provider.add_span_processor(RunAttributes())  # part 3 below
    tracer_provider.add_span_processor(ExperimentAttributes())  # part 4 below
    if pydantic_ai:
        instrument_pydantic_ai(tracer_provider)
    for exporter in make_exporters(settings):
        tracer_provider.add_span_processor(BatchSpanProcessor(exporter))

    # Make this the provider that trace.get_tracer() (below) uses.
    trace.set_tracer_provider(tracer_provider)

    if not pydantic_ai:
        # Automatically create a span for every call made with the OpenAI
        # client. These spans hold the prompt, the model's reply, the tool
        # definitions and the token counts, without us writing any tracing
        # code for them.
        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

    _tracer_provider = tracer_provider
    return tracer_provider


def instrument_pydantic_ai(tracer_provider: TracerProvider) -> None:
    """Trace Pydantic AI agents with the framework's own spans.

    Pydantic AI records each agent run and each model request itself, in
    OpenTelemetry's standard "gen_ai" names. Langfuse reads those directly;
    Phoenix needs OpenInference names, so a processor from OpenInference adds
    them to each span as it ends. The OpenAI instrumentor stays off, because
    Pydantic AI calls the OpenAI client inside its own model-request spans,
    and every call would otherwise be recorded twice.

    The imports are here, not at the top, because these packages are only in
    the "pydantic-ai" dependency group.
    """
    from openinference.instrumentation.pydantic_ai import OpenInferenceSpanProcessor
    from pydantic_ai import Agent
    from pydantic_ai.models.instrumented import InstrumentationSettings

    tracer_provider.add_span_processor(OpenInferenceSpanProcessor())
    Agent.instrument_all(InstrumentationSettings(tracer_provider=tracer_provider))


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
def span(name: str, kind: SpanKind, input_value: str | None = None) -> Generator[Span]:
    """Start a span of any kind. The helpers below are shortcuts for common ones."""
    with tracer.start_as_current_span(name) as current:
        current.set_attribute(SPAN_KIND, kind)
        if input_value is not None:
            current.set_attribute(INPUT, input_value)
        yield current


@contextmanager
def agent_span(prompt: str) -> Generator[Span]:
    """Span for one whole agent run, from the user's prompt to the final answer."""
    with span("agent", SpanKind.AGENT, prompt) as current:
        yield current


@contextmanager
def turn_span(number: int) -> Generator[Span]:
    """Span for one trip to the model and back, including the tools it asked for."""
    with span(f"turn {number}", SpanKind.CHAIN) as current:
        yield current


@contextmanager
def tool_span(name: str, arguments: str) -> Generator[Span]:
    """Span for one tool call. The arguments are the JSON text the model sent."""
    with span(name, SpanKind.TOOL, arguments) as current:
        current.set_attribute(TOOL_NAME, name)
        current.set_attribute(INPUT_MIME_TYPE, "application/json")
        yield current


def set_output(current: Span, value: str) -> None:
    """Record what a step produced, so the tools can show it next to the input."""
    current.set_attribute(OUTPUT, value)


# ---------------------------------------------------------------------------
# 3. Runs: grouping the traces of one run of a program
#
# Each agent call, and each evaluation case, is a trace of its own. Inside
# traced_run(), every span gets the same session ID, so Phoenix and Langfuse
# show the run's traces together on their Sessions pages. Each span also gets
# the user, tags and metadata (the model and git commit), which both tools
# can filter by: "every run with gemma", or "everything since commit abc123".
#
# using_attributes() comes from OpenInference. It puts these values in the
# current context, and RunAttributes copies them onto every span as it
# starts, whoever creates it: span() above, the OpenAI instrumentor, or a
# framework such as Pydantic AI.
# ---------------------------------------------------------------------------


class RunAttributes(SpanProcessor):
    """Copies traced_run()'s session, user, tags and metadata onto every new span."""

    def on_start(self, span: SdkSpan, parent_context: Context | None = None) -> None:
        span.set_attributes(dict(get_attributes_from_context()))


@dataclass
class Run:
    """What traced_run() attached to the run's spans."""

    session_id: str
    user_id: str
    tags: list[str]
    metadata: dict[str, str]


def git_commit() -> str:
    """The current commit, such as "8196f8a", or "8196f8a-dirty" if files were
    changed since. "unknown" if this isn't a git repository."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except OSError, subprocess.CalledProcessError:  # no git, or not a repository
        return "unknown"
    return f"{commit}-dirty" if changes else commit


@contextmanager
def traced_run(
    name: str, tags: list[str] | None = None, settings: Settings | None = None
) -> Generator[Run]:
    """Group every trace made inside this block into one session.

    with traced_run("02_evaluation", tags=["eval"]) as run:
        ...                                  # every span here is labelled
    print(run.session_id)                    # "02_evaluation-20261023-061005"
    """
    if settings is None:
        settings = Settings()
    run = Run(
        session_id=f"{name}-{datetime.now():%Y%m%d-%H%M%S}",
        user_id=settings.trace_user_id,
        tags=[name, *(tags or [])],
        metadata={"model": settings.llm_model, "git_commit": git_commit()},
    )
    with using_attributes(
        session_id=run.session_id, user_id=run.user_id, tags=run.tags, metadata=run.metadata
    ):
        yield run


# ---------------------------------------------------------------------------
# 4. Experiments: an evaluation run on a Langfuse dataset
#
# Langfuse 4 learns that a trace is one case of an experiment from labels on
# its spans: every span of the case's trace, including the ones the OpenAI
# instrumentor makes, needs the experiment's IDs and the case's dataset item
# ID. (Phoenix learns about experiments through its API instead; see
# datasets.py.)
#
# OpenTelemetry's "baggage" carries values to everything that runs inside a
# block, and ExperimentAttributes copies them onto each span as it starts.
# Each case's top span (its "root") also gets the item labels directly.
# ---------------------------------------------------------------------------

EXPERIMENT_PREFIX = "langfuse.experiment."


class ExperimentAttributes(SpanProcessor):
    """Copies the Langfuse experiment labels in the baggage onto every new span."""

    def on_start(self, span: SdkSpan, parent_context: Context | None = None) -> None:
        for key, value in baggage.get_all(parent_context).items():
            if key.startswith(EXPERIMENT_PREFIX):
                span.set_attribute(key, str(value))


@contextmanager
def with_baggage(values: dict[str, str]) -> Generator[None]:
    """Put values in the baggage for everything that runs inside this block."""
    current = context.get_current()
    for key, value in values.items():
        current = baggage.set_baggage(key, value, current)
    token = context.attach(current)
    try:
        yield
    finally:
        context.detach(token)


@contextmanager
def experiment_context(experiment_name: str, dataset_id: str) -> Generator[None]:
    """Mark every trace started inside this block as part of one experiment."""
    with with_baggage(
        {
            f"{EXPERIMENT_PREFIX}id": experiment_name,
            f"{EXPERIMENT_PREFIX}name": experiment_name,
            f"{EXPERIMENT_PREFIX}dataset.id": dataset_id,
        }
    ):
        yield


@contextmanager
def experiment_item(root: Span, item_id: str, expected_output: str) -> Generator[None]:
    """Mark root, a case's top span, and every span under it as one dataset item."""
    item = {
        f"{EXPERIMENT_PREFIX}item.id": item_id,
        # Langfuse needs the root span's own ID, as 16 hex characters.
        f"{EXPERIMENT_PREFIX}item.root_observation_id": format(
            root.get_span_context().span_id, "016x"
        ),
    }
    root.set_attributes(item)
    root.set_attribute(f"{EXPERIMENT_PREFIX}item.expected_output", expected_output)
    with with_baggage(item):
        yield
