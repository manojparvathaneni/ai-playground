"""Keep an evaluation's cases as a dataset in Phoenix and Langfuse, and
record each run of it as an experiment on that dataset.

Scores on traces (scores.py) say how each case did in one run. Datasets and
experiments add the comparison: both tools list every run (experiment) of a
dataset, and can show two runs side by side, case by case. That's how you
check whether a change, such as a new prompt or model, helped.

  sync_dataset()       creates the dataset in both tools, or updates it to
                       match the cases. Call it before each run.
  record_experiment()  records a finished run in Phoenix: one run per case,
                       linked to its trace, with its grade.

Langfuse needs no step after the run: its experiment is recorded from labels
on the spans themselves (tracing.py, part 4), and the scores from scores.py
show up on it.

Problems are printed rather than raised, as in scores.py, so they never hide
the evaluation's own results.
"""

from datetime import datetime
from typing import Any
from urllib.parse import quote

from ai_playground.config import Settings
from ai_playground.evals import Case, Dataset, Experiment, Result
from ai_playground.http_json import get_json, post_json
from ai_playground.scores import SCORE_NAME
from ai_playground.tracing import langfuse_auth_header

# ---------------------------------------------------------------------------
# The dataset
# ---------------------------------------------------------------------------


def phoenix_dataset(dataset: Dataset) -> dict[str, Any]:
    """The request body for Phoenix's dataset upload.

    "update" creates the dataset if it doesn't exist, and otherwise makes it
    match these cases: adding, changing and removing examples by their ID.
    Phoenix keeps a new version of the dataset each time something changes.
    """
    return {
        "action": "update",
        "name": dataset.name,
        "inputs": [{"question": case.question} for case in dataset.cases],
        "outputs": [case.expected for case in dataset.cases],
        "metadata": [{"case": case.name} for case in dataset.cases],
        "example_ids": [dataset.item_id(case) for case in dataset.cases],
    }


def langfuse_item(dataset: Dataset, case: Case) -> dict[str, Any]:
    """The request body for one Langfuse dataset item. Sending one with an
    existing ID updates that item."""
    return {
        "datasetName": dataset.name,
        "id": dataset.item_id(case),
        "input": {"question": case.question},
        "expectedOutput": case.expected,
        "metadata": {"case": case.name},
    }


def sync_dataset(name: str, cases: list[Case], settings: Settings | None = None) -> Dataset:
    """Create the dataset in both tools, or update it to match the cases.

    Returns the dataset with the ID each tool gave it, or None for a tool
    that couldn't be reached.
    """
    if settings is None:
        settings = Settings()
    dataset = Dataset(name, cases)

    try:
        reply = post_json(
            f"{settings.phoenix_collector_endpoint}/v1/datasets/upload?sync=true",
            phoenix_dataset(dataset),
        )
        dataset.phoenix_id = reply["data"]["dataset_id"]
    except Exception as error:
        print(f"Couldn't update the dataset in Phoenix: {error}")

    if settings.langfuse_public_key and settings.langfuse_secret_key:
        try:
            dataset.langfuse_id = sync_langfuse_dataset(
                dataset, settings, settings.langfuse_public_key, settings.langfuse_secret_key
            )
        except Exception as error:
            print(f"Couldn't update the dataset in Langfuse: {error}")

    return dataset


def sync_langfuse_dataset(
    dataset: Dataset, settings: Settings, public_key: str, secret_key: str
) -> str:
    """Make Langfuse's copy of the dataset match its cases. Returns its ID."""
    api = f"{settings.langfuse_host}/api/public"
    auth = {"Authorization": langfuse_auth_header(public_key, secret_key)}

    # Creating a dataset that already exists just returns it.
    reply = post_json(f"{api}/v2/datasets", {"name": dataset.name}, auth)

    for case in dataset.cases:
        post_json(f"{api}/dataset-items", langfuse_item(dataset, case), auth)

    # Archive items whose case was removed. Archived items stay visible in
    # past experiments, but new runs leave them out.
    wanted = {dataset.item_id(case) for case in dataset.cases}
    items = get_json(f"{api}/dataset-items?datasetName={quote(dataset.name)}&limit=100", auth)
    for item in items["data"]:
        if item["id"] not in wanted and item["status"] == "ACTIVE":
            archive = {"datasetName": dataset.name, "id": item["id"], "status": "ARCHIVED"}
            post_json(f"{api}/dataset-items", archive, auth)

    return reply["id"]


# ---------------------------------------------------------------------------
# The experiment (Phoenix)
# ---------------------------------------------------------------------------


def phoenix_run(example_node_id: str, result: Result) -> dict[str, Any]:
    """The request body for one experiment run: one case's answer.

    Phoenix wants the example's own internal ID here (its "node_id"), not the
    ID we gave it.
    """
    return {
        "dataset_example_id": example_node_id,
        "output": {"answer": result.answer},
        "repetition_number": 1,
        "start_time": iso(result.started_at),
        "end_time": iso(result.ended_at),
        "trace_id": result.trace_id,
        # Phoenix counts a run with an error as failed, not successful.
        "error": result.error,
    }


def phoenix_evaluation(run_id: str, result: Result) -> dict[str, Any]:
    """The request body for the grade of one experiment run."""
    return {
        "experiment_run_id": run_id,
        "name": SCORE_NAME,
        "annotator_kind": "CODE",  # graded by code, not by a person or an LLM
        "start_time": iso(result.ended_at),
        "end_time": iso(result.ended_at),
        "result": {
            "label": "pass" if result.grade.passed else "fail",
            "score": 1.0 if result.grade.passed else 0.0,
            "explanation": result.grade.explanation,
        },
    }


def iso(moment: datetime | None) -> str:
    """A time in the ISO 8601 form the APIs expect, such as 2026-10-23T06:15:42+00:00."""
    return (moment or datetime.now()).astimezone().isoformat()


def record_experiment(
    experiment: Experiment, results: list[Result], settings: Settings | None = None
) -> None:
    """Record a finished run as an experiment on the dataset, in Phoenix."""
    if settings is None:
        settings = Settings()
    dataset = experiment.dataset
    if dataset.phoenix_id is None:
        return  # sync_dataset() couldn't create it, and has said so

    api = f"{settings.phoenix_collector_endpoint}/v1"
    try:
        # Each example's internal ID, looked up by the ID we gave it.
        examples = get_json(f"{api}/datasets/{dataset.phoenix_id}/examples")["data"]["examples"]
        node_ids = {example["id"]: example["node_id"] for example in examples}

        reply = post_json(
            f"{api}/datasets/{dataset.phoenix_id}/experiments",
            {"name": experiment.name, "metadata": experiment.metadata},
        )
        experiment_id = reply["data"]["id"]
        for result in results:
            node_id = node_ids[dataset.item_id(result.case)]
            run = post_json(f"{api}/experiments/{experiment_id}/runs", phoenix_run(node_id, result))
            post_json(
                f"{api}/experiment_evaluations", phoenix_evaluation(run["data"]["id"], result)
            )
    except Exception as error:
        print(f"Couldn't record the experiment in Phoenix: {error}")
