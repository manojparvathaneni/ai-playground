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
  2. agent_span(), turn_span() and tool_span() create our own spans, so the
     rest of the code never needs to know attribute names.
"""

import base64
from contextlib import contextmanager
import os

from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


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

def phoenix_exporter():
    """Exporter for a Phoenix server; by default the local one on port 6006."""
    endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
    return OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")


def langfuse_exporter():
    """Exporter for Langfuse; by default the local one on port 3000."""
    host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
    public_key = os.environ["LANGFUSE_PUBLIC_KEY"]
    secret_key = os.environ["LANGFUSE_SECRET_KEY"]

    # Langfuse checks the keys with "Basic" authentication: the two keys
    # joined by a colon, then base64-encoded.
    credentials = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    return OTLPSpanExporter(
        endpoint=f"{host}/api/public/otel/v1/traces",
        headers={"Authorization": f"Basic {credentials}"},
    )


def setup_tracing(project_name: str = "agentic-ai-course"):
    """Send traces to Phoenix, and also to Langfuse if its keys are set.

    Call this once, at the start of the program, before calling the model.
    """
    # The resource describes who is sending the spans. Phoenix uses this
    # attribute to group traces into a project.
    resource = Resource.create({"openinference.project.name": project_name})
    tracer_provider = TracerProvider(resource=resource)

    tracer_provider.add_span_processor(BatchSpanProcessor(phoenix_exporter()))
    if "LANGFUSE_PUBLIC_KEY" in os.environ:
        tracer_provider.add_span_processor(BatchSpanProcessor(langfuse_exporter()))

    # Make this the provider that trace.get_tracer() (below) uses.
    trace.set_tracer_provider(tracer_provider)

    # Automatically create a span for every call made with the OpenAI client.
    # These spans hold the prompt, the model's reply, the tool definitions and
    # the token counts, without us writing any tracing code for them.
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)


# ---------------------------------------------------------------------------
# 2. Our own spans
#
# Attribute names come from OpenInference, the naming scheme Phoenix uses.
# Phoenix reads them to label each span and show its input and output.
# Because they are only written here, changing scheme (for example to
# OpenTelemetry's "gen_ai.*" names) means changing only this file.
#
# If setup_tracing() was never called, these spans do nothing, so the agent
# code also works without Phoenix (for example in the tests).
# ---------------------------------------------------------------------------

SPAN_KIND = "openinference.span.kind"
INPUT = "input.value"
OUTPUT = "output.value"
TOOL_NAME = "tool.name"

tracer = trace.get_tracer(__name__)


# @contextmanager turns a function into something you can use with "with".
# The code before "yield" runs when the "with" block starts, and the span
# ends when the "with" block ends.

@contextmanager
def agent_span(prompt: str):
    """Span for one whole agent run, from the user's prompt to the final answer."""
    with tracer.start_as_current_span("agent") as span:
        span.set_attribute(SPAN_KIND, "AGENT")
        span.set_attribute(INPUT, prompt)
        yield span


@contextmanager
def turn_span(number: int):
    """Span for one trip to the model and back, including the tools it asked for."""
    with tracer.start_as_current_span(f"turn {number}") as span:
        span.set_attribute(SPAN_KIND, "CHAIN")
        yield span


@contextmanager
def tool_span(name: str, arguments: str):
    """Span for one tool call. The arguments are the JSON text the model sent."""
    with tracer.start_as_current_span(name) as span:
        span.set_attribute(SPAN_KIND, "TOOL")
        span.set_attribute(TOOL_NAME, name)
        span.set_attribute(INPUT, arguments)
        yield span


def set_output(span, value: str):
    """Record what a step produced, so Phoenix can show it next to the input."""
    span.set_attribute(OUTPUT, value)
