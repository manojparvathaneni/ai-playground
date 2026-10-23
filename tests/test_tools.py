"""Tests for turning Python functions into tool descriptions (ai_playground/tools.py)."""

import pytest

from ai_playground.tools import get_current_time, get_current_time_in_timezone, make_tool_schema


def test_schema_for_function_without_parameters():
    assert make_tool_schema(get_current_time) == {
        "type": "function",
        "name": "get_current_time",
        "description": "Get the current local time.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    }


def test_schema_for_function_with_parameters():
    schema = make_tool_schema(get_current_time_in_timezone)

    assert schema["parameters"] == {
        "type": "object",
        "properties": {"timezone": {"type": "string"}},
        "required": ["timezone"],
        "additionalProperties": False,
    }


def test_each_supported_type():
    def example(text: str, count: int, ratio: float, flag: bool) -> str:
        """Example."""
        return ""

    parameters = make_tool_schema(example)["parameters"]
    assert parameters is not None
    properties = parameters["properties"]

    assert properties == {
        "text": {"type": "string"},
        "count": {"type": "integer"},
        "ratio": {"type": "number"},
        "flag": {"type": "boolean"},
    }


def test_multi_line_docstring_is_cleaned_up():
    def example() -> str:
        """First line.

        More detail.
        """
        return ""

    assert make_tool_schema(example)["description"] == "First line.\n\nMore detail."


@pytest.mark.parametrize("annotation", [list, dict, None])
def test_unsupported_parameter_type_raises(annotation):
    def example(value):
        return ""

    example.__annotations__["value"] = annotation

    with pytest.raises(TypeError, match="parameter 'value' needs a type hint"):
        make_tool_schema(example)


def test_missing_type_hint_raises():
    def example(value):
        return ""

    with pytest.raises(TypeError, match="parameter 'value' needs a type hint"):
        make_tool_schema(example)
