from copy import deepcopy
import json
import unittest
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfoNotFoundError

from openai.types.responses import Response

from ai_playground.llm import call_llm
from ai_playground.tools import get_current_time, get_current_time_in_timezone


def response_with(*output):
    return Response.model_validate({
        "id": "resp_test",
        "created_at": 0,
        "model": "test-model",
        "object": "response",
        "output": list(output),
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    })


def function_call(name, arguments="{}", call_id="call_test"):
    return {
        "id": f"fc_{call_id}",
        "type": "function_call",
        "name": name,
        "arguments": arguments,
        "call_id": call_id,
    }


FINAL = response_with({
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "Done", "annotations": []}],
})


class CallLLMTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.responses = []
        self.client = MagicMock()
        self.client.responses.create.side_effect = self.create_response
        self.client.__enter__.return_value = self.client
        patcher = patch(
            "ai_playground.llm.OpenAI",
            return_value=self.client,
        )
        self.constructor = patcher.start()
        self.addCleanup(patcher.stop)

    def create_response(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        return self.responses.pop(0)

    def test_no_tools_returns_response(self):
        self.responses = [FINAL]
        result = call_llm("Hello", base_url="http://example.test/v1", model="test-model")
        self.assertIs(result, FINAL)
        self.assertEqual(result.output_text, "Done")
        self.assertEqual(self.requests, [{
            "model": "test-model",
            "input": [{"role": "user", "content": "Hello"}],
            "tools": [],
        }])
        self.constructor.assert_called_once_with(
            base_url="http://example.test/v1", api_key="YOUR_API_KEY"
        )
        self.client.__exit__.assert_called_once()

    def test_time_tools_have_serializable_responses_schemas(self):
        self.responses = [FINAL]
        call_llm("Time?", tools=[get_current_time, get_current_time_in_timezone])
        schemas = self.requests[0]["tools"]
        json.dumps(self.requests[0])
        self.assertEqual(schemas[0]["type"], "function")
        self.assertEqual(schemas[0]["parameters"]["properties"], {})
        self.assertEqual(schemas[0]["parameters"]["required"], [])
        self.assertEqual(schemas[1]["name"], "get_current_time_in_timezone")
        self.assertEqual(schemas[1]["parameters"], {
            "type": "object",
            "properties": {"timezone": {"type": "string"}},
            "required": ["timezone"],
            "additionalProperties": False,
        })

    def test_parallel_and_sequential_calls_preserve_history(self):
        reasoning = {"id": "rs_test", "type": "reasoning", "summary": []}
        first = response_with(
            reasoning,
            function_call("get_current_time", call_id="local"),
            function_call("get_current_time_in_timezone", '{"timezone": "UTC"}', "utc"),
        )
        second = response_with(
            function_call("get_current_time_in_timezone", '{"timezone": "Asia/Tokyo"}', "tokyo")
        )
        self.responses = [first, second, FINAL]
        with patch("ai_playground.tools.datetime") as datetime:
            datetime.now.return_value.strftime.side_effect = ["local time", "UTC time", "Tokyo time"]
            self.assertIs(call_llm(
                "Time?", tools=[get_current_time, get_current_time_in_timezone]
            ), FINAL)
            self.assertEqual(datetime.now.call_count, 3)
            self.assertEqual(str(datetime.now.call_args_list[1].args[0]), "UTC")
            self.assertEqual(str(datetime.now.call_args_list[2].args[0]), "Asia/Tokyo")
        self.assertEqual(len(self.requests), 3)
        history = self.requests[1]["input"]
        self.assertEqual(history[1:4], [
            item.model_dump(exclude_none=True) for item in first.output
        ])
        self.assertEqual(history[4:], [
            {"type": "function_call_output", "call_id": "local", "output": "local time"},
            {"type": "function_call_output", "call_id": "utc", "output": "UTC time"},
        ])
        self.assertEqual(self.requests[2]["input"][:len(history)], history)
        self.assertEqual(self.requests[2]["input"][-1]["call_id"], "tokyo")
        self.assertEqual(self.requests[0]["tools"][0]["parameters"]["required"], [])
        self.assertTrue(self.requests[0]["tools"][0]["strict"])
        for request in self.requests:
            json.dumps(request)

    def test_registered_custom_tool_result_is_json_encoded(self):
        def count() -> dict:
            return {"total": 3}

        self.responses = [response_with(function_call("count")), FINAL]
        call_llm("Count", tools=[count])
        self.assertEqual(
            json.loads(self.requests[1]["input"][-1]["output"]), {"total": 3}
        )

    def test_unknown_tool_raises(self):
        self.responses = [response_with(function_call("unknown"))]
        with self.assertRaisesRegex(ValueError, "unknown tool: unknown"):
            call_llm("Time?")

    def test_invalid_arguments_raise(self):
        for arguments in ("not json", "[]", "null", '{"unexpected": true}'):
            with self.subTest(arguments=arguments):
                self.responses = [response_with(function_call("get_current_time", arguments))]
                with self.assertRaises((ValueError, TypeError)):
                    call_llm("Time?", tools=[get_current_time])

    def test_tool_errors_propagate(self):
        self.responses = [response_with(function_call(
            "get_current_time_in_timezone", '{"timezone": "Invalid/Timezone"}'
        ))]
        with self.assertRaises(ZoneInfoNotFoundError):
            call_llm("Time?", tools=[get_current_time_in_timezone])
        self.client.__exit__.assert_called_once()

    def test_api_errors_propagate(self):
        self.client.responses.create.side_effect = RuntimeError("API failed")
        with self.assertRaisesRegex(RuntimeError, "API failed"):
            call_llm("Hello")

    def test_unsupported_parameter_type_raises_before_request(self):
        def unsupported(items: list):
            return "result"

        with self.assertRaises(KeyError):
            call_llm("Run", tools=[unsupported])
        self.constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
