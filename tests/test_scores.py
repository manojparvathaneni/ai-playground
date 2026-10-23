"""Tests for sending eval scores to Phoenix and Langfuse (ai_playground/scores.py)."""

import urllib.error
from email.message import Message
from unittest.mock import patch

from ai_playground.evals import Case, Grade, Result
from ai_playground.scores import langfuse_scores, phoenix_annotations, publish_scores

TRACE_ID = "0123456789abcdef0123456789abcdef"
SPAN_ID = "0123456789abcdef"


def passed_result(trace_id=TRACE_ID):
    return Result(Case("utc", "What time is it in UTC?"), "04:00", Grade(True, "correct"), trace_id)


def failed_result():
    return Result(Case("local", "Time here?"), "?", Grade(False, "no times"), TRACE_ID, SPAN_ID)


def test_phoenix_annotations():
    assert phoenix_annotations([passed_result(), failed_result()]) == {
        "data": [
            {
                "trace_id": TRACE_ID,
                "name": "correct",
                "annotator_kind": "CODE",
                "result": {"label": "pass", "score": 1.0, "explanation": "correct"},
            },
            {
                "trace_id": TRACE_ID,
                "name": "correct",
                "annotator_kind": "CODE",
                "result": {"label": "fail", "score": 0.0, "explanation": "no times"},
            },
        ]
    }


def test_langfuse_scores():
    assert langfuse_scores([failed_result()]) == [
        {
            "traceId": TRACE_ID,
            "observationId": SPAN_ID,
            "name": "correct",
            "value": 0,
            "dataType": "BOOLEAN",
            "comment": "no times",
        }
    ]


def test_publish_sends_to_phoenix_only_without_langfuse_keys(settings):
    with patch("ai_playground.scores.post_json") as post_json:
        publish_scores([passed_result()], settings)

    (call,) = post_json.call_args_list
    assert call.args[0] == "http://localhost:6006/v1/trace_annotations?sync=true"


def test_publish_sends_one_langfuse_score_per_result(settings):
    with_keys = settings.model_copy(
        update={"langfuse_public_key": "pk", "langfuse_secret_key": "sk"}
    )

    with patch("ai_playground.scores.post_json") as post_json:
        publish_scores([passed_result(), failed_result()], with_keys)

    urls = [call.args[0] for call in post_json.call_args_list]
    assert urls == [
        "http://localhost:6006/v1/trace_annotations?sync=true",
        "http://localhost:3000/api/public/scores",
        "http://localhost:3000/api/public/scores",
    ]
    assert post_json.call_args_list[1].args[2] == {"Authorization": "Basic cGs6c2s="}


def test_results_without_a_trace_are_skipped(settings):
    with patch("ai_playground.scores.post_json") as post_json:
        publish_scores([passed_result(trace_id=None)], settings)

    post_json.assert_not_called()


def test_a_tool_that_cant_be_reached_doesnt_raise(settings, capsys):
    with patch("ai_playground.scores.post_json", side_effect=OSError("connection refused")):
        publish_scores([passed_result()], settings)

    assert "Couldn't send scores to Phoenix: connection refused" in capsys.readouterr().out


def http_error(code):
    return urllib.error.HTTPError("http://phoenix.test", code, "error", Message(), None)


def test_phoenix_is_asked_again_until_it_has_saved_the_trace(settings):
    replies = [http_error(404), http_error(404), None]  # not saved yet, twice
    with (
        patch("ai_playground.scores.post_json", side_effect=replies) as post_json,
        patch("ai_playground.scores.time.sleep") as sleep,
    ):
        publish_scores([passed_result()], settings)

    assert post_json.call_count == 3
    assert sleep.call_count == 2


def test_other_phoenix_errors_are_not_retried(settings, capsys):
    with patch("ai_playground.scores.post_json", side_effect=http_error(500)) as post_json:
        publish_scores([passed_result()], settings)

    assert post_json.call_count == 1
    assert "Couldn't send scores to Phoenix: HTTP Error 500" in capsys.readouterr().out
