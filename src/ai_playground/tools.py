"""Tools the model can call, and how they are described to it.

To give the model a new tool, write a function with a docstring and type
hints, and pass it in call_llm(..., tools=[...]). make_tool_schema() turns it
into the description the model reads.
"""

import inspect
from collections.abc import Callable
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from openai.types.responses import FunctionToolParam

# A tool is any Python function. Callable[..., Any] means "a function that
# takes any arguments and returns anything".
Tool = Callable[..., Any]


# ---------------------------------------------------------------------------
# Example tools: plain Python functions the model is allowed to call.
# The docstring is what the model reads to decide when to use each tool.
# ---------------------------------------------------------------------------


def format_time(moment: datetime) -> str:
    """Write a time on both clocks, such as "01:13:00 (1:13 AM)".

    With only "01:13:00", small models sometimes say "1:13 PM". Spelling out
    AM or PM leaves them nothing to guess.
    """
    twelve_hour = moment.strftime("%I:%M %p").lstrip("0")  # "01:13 AM" -> "1:13 AM"
    return f"{moment.strftime('%H:%M:%S')} ({twelve_hour})"


def get_current_time() -> str:
    """Get the current time where the user is, in their own timezone.

    Use this when the user asks for the time "here", the local time, or the
    time without naming a place.
    """
    return format_time(datetime.now())


def get_current_time_in_timezone(timezone: str) -> str:
    """Get the current time in an IANA timezone, such as UTC or Asia/Tokyo."""
    return format_time(datetime.now(ZoneInfo(timezone)))


# ---------------------------------------------------------------------------
# Describing tools to the model
# ---------------------------------------------------------------------------

# How each Python type is written in JSON Schema.
JSON_TYPES: dict[type, str] = {str: "string", int: "integer", float: "number", bool: "boolean"}


def make_tool_schema(tool: Tool) -> FunctionToolParam:
    """Describe a Python function in the JSON format the model understands.

    For example, for this function:

        def get_current_time_in_timezone(timezone: str) -> str:
            \"\"\"Get the current time in an IANA timezone, such as UTC or Asia/Tokyo.\"\"\"

    it returns:

        {
            "type": "function",
            "name": "get_current_time_in_timezone",
            "description": "Get the current time in an IANA timezone, such as UTC or Asia/Tokyo.",
            "parameters": {
                "type": "object",
                "properties": {"timezone": {"type": "string"}},
                "required": ["timezone"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    """
    # inspect.signature() lets Python look at a function's own definition.
    # Its .parameters holds one entry per argument. For the example above
    # that is just "timezone", and parameter.annotation is its type hint, str.
    parameters = inspect.signature(tool).parameters

    properties: dict[str, dict[str, str]] = {}  # each argument's name and JSON type
    required: list[str] = []  # arguments the model must fill in
    for name, parameter in parameters.items():
        python_type = parameter.annotation  # e.g. str
        if python_type not in JSON_TYPES:
            raise TypeError(
                f"Tool {tool.__name__}: parameter '{name}' needs a type hint of "
                f"str, int, float or bool, not {python_type!r}"
            )
        json_type = JSON_TYPES[python_type]  # e.g. "string"
        properties[name] = {"type": json_type}
        required.append(name)

    return {
        "type": "function",
        "name": tool.__name__,
        "description": inspect.getdoc(tool) or "",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        "strict": True,
    }


# ---------------------------------------------------------------------------
# For comparison: the manual way (not used)
#
# Without make_tool_schema, we would write every schema by hand, like this:
#
# TOOL_SCHEMAS = {
#     "get_current_time": {
#         "type": "function",
#         "name": "get_current_time",
#         "description": "Get the current local time.",
#         "parameters": {
#             "type": "object",
#             "properties": {},
#             "required": [],
#             "additionalProperties": False,
#         },
#         "strict": True,
#     },
#     "get_current_time_in_timezone": {
#         "type": "function",
#         "name": "get_current_time_in_timezone",
#         "description": "Get the current time in an IANA timezone, such as UTC or Asia/Tokyo.",
#         "parameters": {
#             "type": "object",
#             "properties": {"timezone": {"type": "string"}},
#             "required": ["timezone"],
#             "additionalProperties": False,
#         },
#         "strict": True,
#     },
# }
#
# and call_llm would look schemas up with TOOL_SCHEMAS[tool.__name__].
#
# Why make_tool_schema is better:
#   1. Less typing. Each tool here needs about 12 lines of schema; the
#      automatic version needs none.
#   2. Nothing to keep in sync. The name, description and arguments are
#      written twice (once in the function, once in the schema). If you
#      rename an argument in the function but forget the schema, the model
#      sends the old name and the tool call crashes.
#   3. Adding a tool is one step. Write the function and pass it in
#      tools=[...]. The manual way also needs a new TOOL_SCHEMAS entry.
#
# The manual way is still worth seeing once: it shows exactly what the model
# receives, and it is what make_tool_schema builds for you.
# ---------------------------------------------------------------------------
