"""The agent loop: ask the model, run the tools it asks for, repeat until it answers."""

import json

from openai import OpenAI

from ai_playground.tools import make_tool_schema
from ai_playground.tracing import agent_span, set_output, tool_span, turn_span


# ---------------------------------------------------------------------------
# The agent loop
# ---------------------------------------------------------------------------

def run_tool(call, tools_by_name: dict) -> dict:
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


def call_llm(prompt: str, base_url="http://localhost:1234/v1", model="", tools=None):
    """Ask the model a question, running any tools it asks for until it answers."""
    if tools is None:
        tools = []

    # Look up tools by name, and describe each one for the model.
    tools_by_name = {}
    tool_schemas = []
    for tool in tools:
        tools_by_name[tool.__name__] = tool
        tool_schemas.append(make_tool_schema(tool))

    # The conversation so far. We send all of it to the model every time.
    history = [{"role": "user", "content": prompt}]

    with agent_span(prompt) as agent:

        with OpenAI(base_url=base_url, api_key="YOUR_API_KEY") as client:
            turn = 0
            while True:
                turn += 1
                with turn_span(turn) as current_turn:

                    response = client.responses.create(
                        model=model, input=history, tools=tool_schemas
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
                        history.append(item.model_dump(exclude_none=True))

                    # Run each requested tool and add its result to the history.
                    for call in tool_calls:
                        history.append(run_tool(call, tools_by_name))
