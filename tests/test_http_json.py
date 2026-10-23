"""Tests for the JSON-over-HTTP helpers (ai_playground/http_json.py)."""

import json
from unittest.mock import MagicMock, patch

from ai_playground.http_json import get_json, post_json


def fake_reply(body: bytes) -> MagicMock:
    reply = MagicMock()
    reply.__enter__.return_value.read.return_value = body
    return reply


def test_post_sends_json_and_returns_the_reply():
    with patch("urllib.request.urlopen", return_value=fake_reply(b'{"id": "1"}')) as urlopen:
        assert post_json("http://example.test/x", {"a": 1}, {"Authorization": "k"}) == {"id": "1"}

    request = urlopen.call_args.args[0]
    assert request.method == "POST"
    assert json.loads(request.data) == {"a": 1}
    assert request.get_header("Authorization") == "k"
    assert request.get_header("Content-type") == "application/json"


def test_get_sends_no_body_and_an_empty_reply_is_none():
    with patch("urllib.request.urlopen", return_value=fake_reply(b"")) as urlopen:
        assert get_json("http://example.test/x") is None

    request = urlopen.call_args.args[0]
    assert request.method == "GET"
    assert request.data is None
