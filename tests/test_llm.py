"""Tests for the agent loop in ai_playground/llm.py."""

import json
from unittest.mock import patch
from zoneinfo import ZoneInfoNotFoundError

import pytest

from ai_playground.llm import TooManyTurnsError, call_llm
from ai_playground.tools import get_current_time, get_current_time_in_timezone
from tests.fakes import function_call, response_with, text_answer


def test_answer_without_tools(settings, model):
    answer = text_answer("Hello back")
    model.replies = [answer]

    result = call_llm("Hello", settings=settings)

    assert result is answer
    assert result.output_text == "Hello back"
    assert model.requests == [
        {
            "model": "test-model",
            "input": [{"role": "user", "content": "Hello"}],
            "tools": [],
        }
    ]


def test_client_uses_settings(settings, model):
    model.replies = [text_answer()]

    call_llm("Hello", settings=settings)

    model.constructor.assert_called_once_with(
        base_url="http://example.test/v1",
        api_key="test-key",
        timeout=5,
        max_retries=0,
    )
    model.client.__exit__.assert_called_once()


def test_tools_are_described_to_the_model(settings, model):
    model.replies = [text_answer()]

    call_llm("Time?", tools=[get_current_time, get_current_time_in_timezone], settings=settings)

    schemas = model.requests[0]["tools"]
    assert [schema["name"] for schema in schemas] == [
        "get_current_time",
        "get_current_time_in_timezone",
    ]
    json.dumps(model.requests[0])  # the request must be plain JSON


def test_tool_results_are_sent_back_over_several_turns(settings, model):
    reasoning = {"id": "rs_test", "type": "reasoning", "summary": []}
    first = response_with(
        reasoning,
        function_call("get_current_time", call_id="local"),
        function_call("get_current_time_in_timezone", '{"timezone": "UTC"}', "utc"),
    )
    second = response_with(
        function_call("get_current_time_in_timezone", '{"timezone": "Asia/Tokyo"}', "tokyo")
    )
    final = text_answer()
    model.replies = [first, second, final]

    with patch("ai_playground.tools.datetime") as datetime:
        datetime.now.return_value.strftime.side_effect = ["local time", "UTC time", "Tokyo time"]
        result = call_llm(
            "Time?", tools=[get_current_time, get_current_time_in_timezone], settings=settings
        )

    assert result is final
    assert len(model.requests) == 3

    # Turn 2 is sent the question, the model's whole first reply (including
    # its reasoning), and both tool results.
    history = model.requests[1]["input"]
    assert history[1:4] == [item.model_dump(exclude_none=True) for item in first.output]
    assert history[4:] == [
        {"type": "function_call_output", "call_id": "local", "output": "local time"},
        {"type": "function_call_output", "call_id": "utc", "output": "UTC time"},
    ]

    # Turn 3 is sent everything from turn 2, plus the new tool result.
    assert model.requests[2]["input"][: len(history)] == history
    assert model.requests[2]["input"][-1]["call_id"] == "tokyo"


def test_non_text_tool_results_are_sent_as_json(settings, model):
    def count() -> dict[str, int]:
        """Count things."""
        return {"total": 3}

    model.replies = [response_with(function_call("count")), text_answer()]

    call_llm("Count", tools=[count], settings=settings)

    assert json.loads(model.requests[1]["input"][-1]["output"]) == {"total": 3}


def test_stops_after_max_turns(settings, model):
    # A model that asks for a tool every time would otherwise loop forever.
    model.replies = [response_with(function_call("get_current_time"))] * 3

    with pytest.raises(TooManyTurnsError, match="after 3 turns"):
        call_llm("Time?", tools=[get_current_time], settings=settings)

    assert len(model.requests) == settings.agent_max_turns


def test_unknown_tool_raises(settings, model):
    model.replies = [response_with(function_call("unknown"))]

    with pytest.raises(ValueError, match="unknown tool: unknown"):
        call_llm("Time?", settings=settings)


@pytest.mark.parametrize("arguments", ["not json", "[]", "null", '{"unexpected": true}'])
def test_invalid_tool_arguments_raise(settings, model, arguments):
    model.replies = [response_with(function_call("get_current_time", arguments))]

    with pytest.raises((ValueError, TypeError)):
        call_llm("Time?", tools=[get_current_time], settings=settings)


def test_tool_errors_propagate(settings, model):
    model.replies = [
        response_with(
            function_call("get_current_time_in_timezone", '{"timezone": "Invalid/Timezone"}')
        )
    ]

    with pytest.raises(ZoneInfoNotFoundError):
        call_llm("Time?", tools=[get_current_time_in_timezone], settings=settings)

    model.client.__exit__.assert_called_once()  # the client is still closed


def test_api_errors_propagate(settings, model):
    model.client.responses.create.side_effect = RuntimeError("API failed")

    with pytest.raises(RuntimeError, match="API failed"):
        call_llm("Hello", settings=settings)
