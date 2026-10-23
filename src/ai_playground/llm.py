"""The agent loop: ask the model, run the tools it asks for, repeat until it answers."""

import json
from typing import cast

from openai import OpenAI
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseInputItemParam,
    ResponseOutputItem,
)
from openai.types.responses.response_input_param import FunctionCallOutput

from ai_playground.config import Settings
from ai_playground.tools import Tool, make_tool_schema
from ai_playground.tracing import agent_span, set_output, tool_span, turn_span


class TooManyTurnsError(RuntimeError):
    """The model was still asking for tools after the maximum number of turns."""


def run_tool(call: ResponseFunctionToolCall, tools_by_name: dict[str, Tool]) -> FunctionCallOutput:
    """Run one tool the model asked for, and return the result to send back."""
    with tool_span(call.name, call.arguments) as span:
        if call.name not in tools_by_name:
            raise ValueError(f"Model requested an unknown tool: {call.name}")

        tool = tools_by_name[call.name]
        arguments = json.loads(call.arguments)
        result = tool(**arguments)

        if not isinstance(result, str):
            result = json.dumps(result)

        set_output(span, result)

    return {
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": result,
    }


def as_input_item(item: ResponseOutputItem) -> ResponseInputItemParam:
    """Turn an item from the model's reply into one we can send back to it.

    The reply holds Python objects, but the history we send must be plain
    dictionaries. They have the same fields, so we convert and tell the type
    checker it's the right shape with cast().
    """
    return cast(ResponseInputItemParam, item.model_dump(exclude_none=True))


def call_llm(
    prompt: str,
    tools: list[Tool] | None = None,
    settings: Settings | None = None,
) -> Response:
    """Ask the model a question, running any tools it asks for until it answers.

    Settings such as the model and server come from Settings(), so from
    environment variables or .env, unless you pass your own.
    """
    if tools is None:
        tools = []
    if settings is None:
        settings = Settings()

    # Look up tools by name, and describe each one for the model.
    tools_by_name = {}
    tool_schemas = []
    for tool in tools:
        tools_by_name[tool.__name__] = tool
        tool_schemas.append(make_tool_schema(tool))

    # The conversation so far. We send all of it to the model every time.
    history: list[ResponseInputItemParam] = [{"role": "user", "content": prompt}]

    client = OpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        # Give up on a request that takes too long, and retry ones that fail
        # for temporary reasons, such as the server being briefly busy.
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )

    with agent_span(prompt) as agent, client:
        for turn in range(1, settings.agent_max_turns + 1):
            with turn_span(turn) as current_turn:
                response = client.responses.create(
                    model=settings.llm_model, input=history, tools=tool_schemas
                )

                # Did the model ask us to run any tools?
                tool_calls = []
                for item in response.output:
                    if item.type == "function_call":
                        tool_calls.append(item)

                # No tool calls means the model has its final answer.
                if not tool_calls:
                    set_output(current_turn, response.output_text)
                    set_output(agent, response.output_text)
                    return response

                # Add the model's reply to the history (including any reasoning items).
                for item in response.output:
                    history.append(as_input_item(item))

                # Run each requested tool and add its result to the history.
                for call in tool_calls:
                    history.append(run_tool(call, tools_by_name))

    raise TooManyTurnsError(
        f"The model was still asking for tools after {settings.agent_max_turns} turns."
    )
