"""Tests for keeping datasets and experiments in Phoenix and Langfuse
(ai_playground/datasets.py). The HTTP calls are faked."""

from datetime import UTC, datetime
from unittest.mock import patch

from ai_playground.datasets import (
    langfuse_item,
    phoenix_dataset,
    phoenix_evaluation,
    phoenix_run,
    record_experiment,
    sync_dataset,
)
from ai_playground.evals import Case, Dataset, Experiment, Grade, Result

CASES = [
    Case("utc", "What time is it in UTC?", {"timezones": ["UTC"]}),
    Case("two cities", "New York and London?", {"timezones": ["America/New_York"]}),
]
START = datetime(2026, 10, 23, 6, 0, 0, tzinfo=UTC)
END = datetime(2026, 10, 23, 6, 0, 5, tzinfo=UTC)


def with_langfuse(settings):
    return settings.model_copy(update={"langfuse_public_key": "pk", "langfuse_secret_key": "sk"})


def test_phoenix_dataset_keeps_the_dataset_in_step_with_the_cases():
    assert phoenix_dataset(Dataset("time", CASES)) == {
        "action": "update",
        "name": "time",
        "inputs": [{"question": "What time is it in UTC?"}, {"question": "New York and London?"}],
        "outputs": [{"timezones": ["UTC"]}, {"timezones": ["America/New_York"]}],
        "metadata": [{"case": "utc"}, {"case": "two cities"}],
        "example_ids": ["time-utc", "time-two-cities"],
    }


def test_langfuse_item():
    assert langfuse_item(Dataset("time", CASES), CASES[1]) == {
        "datasetName": "time",
        "id": "time-two-cities",
        "input": {"question": "New York and London?"},
        "expectedOutput": {"timezones": ["America/New_York"]},
        "metadata": {"case": "two cities"},
    }


def fake_post(url, body, headers=None):
    """Replies like the two tools do, for the calls sync_dataset() makes."""
    if url.endswith("/v1/datasets/upload?sync=true"):
        return {"data": {"dataset_id": "phoenix-id"}}
    if url.endswith("/api/public/v2/datasets"):
        return {"id": "langfuse-id"}
    return {}


def test_sync_dataset_updates_both_tools_and_archives_removed_cases(settings):
    stored_items = {
        "data": [
            {"id": "time-utc", "status": "ACTIVE"},
            {"id": "time-removed", "status": "ACTIVE"},  # its case was deleted
            {"id": "time-old", "status": "ARCHIVED"},  # already archived
        ]
    }
    with (
        patch("ai_playground.datasets.post_json", side_effect=fake_post) as post_json,
        patch("ai_playground.datasets.get_json", return_value=stored_items),
    ):
        dataset = sync_dataset("time", CASES, with_langfuse(settings))

    assert (dataset.phoenix_id, dataset.langfuse_id) == ("phoenix-id", "langfuse-id")
    sent_items = [
        c.args[1] for c in post_json.call_args_list if c.args[0].endswith("dataset-items")
    ]
    assert [item["id"] for item in sent_items] == ["time-utc", "time-two-cities", "time-removed"]
    assert sent_items[-1]["status"] == "ARCHIVED"


def test_sync_dataset_skips_langfuse_without_keys(settings):
    with patch("ai_playground.datasets.post_json", side_effect=fake_post) as post_json:
        dataset = sync_dataset("time", CASES, settings)

    assert (dataset.phoenix_id, dataset.langfuse_id) == ("phoenix-id", None)
    assert post_json.call_count == 1  # Phoenix only


def test_a_tool_that_cant_be_reached_is_reported_not_raised(settings, capsys):
    with patch("ai_playground.datasets.post_json", side_effect=OSError("connection refused")):
        dataset = sync_dataset("time", CASES, with_langfuse(settings))

    assert (dataset.phoenix_id, dataset.langfuse_id) == (None, None)
    output = capsys.readouterr().out
    assert "Couldn't update the dataset in Phoenix: connection refused" in output
    assert "Couldn't update the dataset in Langfuse: connection refused" in output


def result(case, passed=True):
    grade = Grade(passed, "ok" if passed else "wrong")
    return Result(case, "an answer", grade, "a" * 32, "b" * 16, START, END)


def test_phoenix_run_and_evaluation():
    utc = result(CASES[0])

    assert phoenix_run("node-1", utc) == {
        "dataset_example_id": "node-1",
        "output": {"answer": "an answer"},
        "repetition_number": 1,
        "start_time": START.astimezone().isoformat(),
        "end_time": END.astimezone().isoformat(),
        "trace_id": "a" * 32,
        "error": None,
    }
    assert phoenix_evaluation("run-1", result(CASES[0], passed=False))["result"] == {
        "label": "fail",
        "score": 0.0,
        "explanation": "wrong",
    }


def test_record_experiment_adds_a_run_and_grade_per_case(settings):
    dataset = Dataset("time", CASES, phoenix_id="phoenix-id")
    examples = {
        "data": {
            "examples": [
                {"id": "time-utc", "node_id": "node-1"},
                {"id": "time-two-cities", "node_id": "node-2"},
            ]
        }
    }
    with (
        patch("ai_playground.datasets.get_json", return_value=examples),
        patch("ai_playground.datasets.post_json", return_value={"data": {"id": "new-id"}}) as post,
    ):
        record_experiment(
            Experiment("run-1", dataset, {"model": "m"}),
            [result(CASES[0]), result(CASES[1])],
            settings,
        )

    urls = [
        c.args[0].removeprefix(settings.phoenix_collector_endpoint) for c in post.call_args_list
    ]
    assert urls == [
        "/v1/datasets/phoenix-id/experiments",
        "/v1/experiments/new-id/runs",
        "/v1/experiment_evaluations",
        "/v1/experiments/new-id/runs",
        "/v1/experiment_evaluations",
    ]
    assert post.call_args_list[0].args[1] == {"name": "run-1", "metadata": {"model": "m"}}
    runs = [c.args[1] for c in post.call_args_list if c.args[0].endswith("/runs")]
    assert [run["dataset_example_id"] for run in runs] == ["node-1", "node-2"]


def test_record_experiment_does_nothing_without_a_phoenix_dataset(settings):
    with patch("ai_playground.datasets.post_json") as post_json:
        record_experiment(Experiment("run-1", Dataset("time", CASES)), [], settings)

    post_json.assert_not_called()
