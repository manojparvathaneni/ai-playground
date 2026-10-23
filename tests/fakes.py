"""Fake model replies for tests, so they run without a model server."""

from copy import deepcopy
from typing import Any
from unittest.mock import MagicMock

from openai.types.responses import Response


def response_with(*output: dict[str, Any]) -> Response:
    """A model reply containing the given output items."""
    return Response.model_validate(
        {
            "id": "resp_test",
            "created_at": 0,
            "model": "test-model",
            "object": "response",
            "output": list(output),
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
    )


def function_call(name: str, arguments: str = "{}", call_id: str = "call_test") -> dict[str, Any]:
    """An output item in which the model asks for a tool."""
    return {
        "id": f"fc_{call_id}",
        "type": "function_call",
        "name": name,
        "arguments": arguments,
        "call_id": call_id,
    }


def text_answer(text: str = "Done") -> Response:
    """A model reply with a final text answer and no tool calls."""
    return response_with(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }
    )


class FakeModel:
    """Stands in for the model server: returns queued replies, records requests."""

    def __init__(self) -> None:
        self.replies: list[Response] = []
        self.requests: list[dict[str, Any]] = []
        # Set by the `model` fixture: the fake OpenAI client, and the mock
        # that replaced the OpenAI class (to check how it was created).
        self.client = MagicMock()
        self.constructor = MagicMock()

    def create(self, **request: Any) -> Response:
        # Copy, because call_llm keeps adding to the same history list.
        self.requests.append(deepcopy(request))
        return self.replies.pop(0)
