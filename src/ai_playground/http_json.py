"""Send and receive JSON over HTTP, for talking to Phoenix's and Langfuse's APIs.

Python's built-in urllib is enough here, so no extra package is needed.
"""

import json
import urllib.request
from typing import Any


def request_json(
    method: str, url: str, body: Any = None, headers: dict[str, str] | None = None
) -> Any:
    """Send a request and return the JSON reply (None if the reply is empty).

    Raises an error if the server rejects the request.
    """
    request = urllib.request.Request(
        url,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        reply = response.read()
    return json.loads(reply) if reply else None


def get_json(url: str, headers: dict[str, str] | None = None) -> Any:
    """GET a URL and return its JSON reply."""
    return request_json("GET", url, headers=headers)


def post_json(url: str, body: Any, headers: dict[str, str] | None = None) -> Any:
    """POST JSON to a URL and return its JSON reply."""
    return request_json("POST", url, body, headers)
