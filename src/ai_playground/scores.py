"""Send evaluation scores to Phoenix and Langfuse, so they appear next to the traces.

Traces record what happened; scores record how good it was. The two tools
keep them separately and receive them through different APIs:

  - Phoenix calls them annotations. Each one has a label ("pass" or "fail"),
    a score and an explanation, and is attached to a trace.
  - Langfuse calls them scores. Each one has a value and a comment, and is
    attached to a trace.

Once sent, you can filter traces by score in both tools (for example, show
only the failures) and see the explanation next to each trace.
"""

import json
import urllib.request
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from ai_playground.config import Settings
from ai_playground.evals import Result
from ai_playground.tracing import langfuse_auth_header

# The name the score appears under in both tools.
SCORE_NAME = "correct"


def post_json(url: str, body: Any, headers: dict[str, str] | None = None) -> None:
    """Send JSON to a URL with an HTTP POST. Raises an error if it's rejected."""
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10):
        pass


def phoenix_annotations(results: list[Result]) -> dict[str, Any]:
    """The request body for Phoenix's trace annotations API."""
    return {
        "data": [
            {
                "trace_id": result.trace_id,
                "name": SCORE_NAME,
                "annotator_kind": "CODE",  # scored by code, not by a person or an LLM
                "result": {
                    "label": "pass" if result.grade.passed else "fail",
                    "score": 1.0 if result.grade.passed else 0.0,
                    "explanation": result.grade.explanation,
                },
            }
            for result in results
        ]
    }


def langfuse_scores(results: list[Result]) -> list[dict[str, Any]]:
    """The request bodies for Langfuse's scores API, one per result."""
    return [
        {
            "traceId": result.trace_id,
            "name": SCORE_NAME,
            "value": 1 if result.grade.passed else 0,
            "dataType": "BOOLEAN",
            "comment": result.grade.explanation,
        }
        for result in results
    ]


def publish_scores(results: list[Result], settings: Settings | None = None) -> None:
    """Attach each result's grade to its trace in Phoenix and in Langfuse.

    Problems sending scores are printed rather than raised, so they never
    hide the evaluation's own results.
    """
    if settings is None:
        settings = Settings()
    results = [result for result in results if result.trace_id is not None]
    if not results:
        return

    # Spans are sent in the background. Send any that are still waiting, so
    # each trace exists in the tools before we attach a score to it.
    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        provider.force_flush()

    try:
        post_json(
            f"{settings.phoenix_collector_endpoint}/v1/trace_annotations?sync=true",
            phoenix_annotations(results),
        )
    except Exception as error:
        print(f"Couldn't send scores to Phoenix: {error}")

    if settings.langfuse_public_key and settings.langfuse_secret_key:
        auth = {
            "Authorization": langfuse_auth_header(
                settings.langfuse_public_key, settings.langfuse_secret_key
            )
        }
        try:
            for score in langfuse_scores(results):
                post_json(f"{settings.langfuse_host}/api/public/scores", score, auth)
        except Exception as error:
            print(f"Couldn't send scores to Langfuse: {error}")
