"""Tests for the spans we create (ai_playground/tracing.py)."""

from unittest.mock import patch

import pytest
from opentelemetry.trace import StatusCode

from ai_playground.llm import call_llm
from ai_playground.tools import get_current_time, get_current_time_in_timezone
from ai_playground.tracing import SpanKind, langfuse_auth_header, make_exporters, span
from tests.fakes import function_call, response_with, text_answer


def tree(finished_spans):
    """The spans as (depth, name, kind) in start order, for easy comparison."""
    by_id = {s.context.span_id: s for s in finished_spans}

    def depth(s):
        return 0 if s.parent is None else 1 + depth(by_id[s.parent.span_id])

    ordered = sorted(finished_spans, key=lambda s: s.start_time)
    return [(depth(s), s.name, s.attributes["openinference.span.kind"]) for s in ordered]


def test_agent_run_produces_the_expected_tree(settings, model, span_exporter):
    model.replies = [
        response_with(
            function_call("get_current_time", call_id="local"),
            function_call("get_current_time_in_timezone", '{"timezone": "UTC"}', "utc"),
        ),
        text_answer("It is noon."),
    ]

    with patch("ai_playground.tools.format_time") as format_time:
        format_time.return_value = "12:00:00"
        call_llm("Time?", tools=[get_current_time, get_current_time_in_timezone], settings=settings)

    # The model calls themselves are traced by the OpenAI instrumentor, which
    # the tests don't switch on, so only our own spans appear here.
    assert tree(span_exporter.get_finished_spans()) == [
        (0, "agent", "AGENT"),
        (1, "turn 1", "CHAIN"),
        (2, "get_current_time", "TOOL"),
        (2, "get_current_time_in_timezone", "TOOL"),
        (1, "turn 2", "CHAIN"),
    ]


def test_spans_record_inputs_and_outputs(settings, model, span_exporter):
    model.replies = [
        response_with(function_call("get_current_time_in_timezone", '{"timezone": "UTC"}')),
        text_answer("It is noon."),
    ]

    with patch("ai_playground.tools.format_time") as format_time:
        format_time.return_value = "12:00:00"
        call_llm("Time?", tools=[get_current_time_in_timezone], settings=settings)

    by_name = {s.name: s.attributes for s in span_exporter.get_finished_spans()}
    assert by_name["agent"]["input.value"] == "Time?"
    assert by_name["agent"]["output.value"] == "It is noon."
    assert by_name["get_current_time_in_timezone"]["input.value"] == '{"timezone": "UTC"}'
    assert by_name["get_current_time_in_timezone"]["output.value"] == "12:00:00"
    assert by_name["get_current_time_in_timezone"]["tool.name"] == "get_current_time_in_timezone"


def test_tool_error_marks_every_span_it_passes_through(settings, model, span_exporter):
    model.replies = [
        response_with(function_call("get_current_time_in_timezone", '{"timezone": "Mars/Olympus"}'))
    ]

    with pytest.raises(Exception):  # noqa: B017 - which error doesn't matter here
        call_llm("Time on Mars?", tools=[get_current_time_in_timezone], settings=settings)

    for finished in span_exporter.get_finished_spans():
        assert finished.status.status_code == StatusCode.ERROR, finished.name
        assert [event.name for event in finished.events] == ["exception"]


def test_span_helper_sets_kind_and_input(span_exporter):
    with span("search docs", SpanKind.RETRIEVER, "what is a span?"):
        pass

    (finished,) = span_exporter.get_finished_spans()
    assert finished.name == "search docs"
    assert finished.attributes["openinference.span.kind"] == "RETRIEVER"
    assert finished.attributes["input.value"] == "what is a span?"


def test_traces_go_to_phoenix_only_without_langfuse_keys(settings):
    assert len(make_exporters(settings)) == 1


def test_traces_go_to_both_tools_with_langfuse_keys(settings):
    with_keys = settings.model_copy(
        update={"langfuse_public_key": "pk", "langfuse_secret_key": "sk"}
    )

    assert len(make_exporters(with_keys)) == 2


def test_langfuse_auth_header():
    # "pk:sk", base64-encoded
    assert langfuse_auth_header("pk", "sk") == "Basic cGs6c2s="
