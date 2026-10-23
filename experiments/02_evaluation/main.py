"""Experiment 02: measure how well the tool-calling agent from experiment 01 answers.

Run it after any change (a different model, prompt or tool) and compare the
pass rate. Failing cases are printed, and every case is traced and scored,
so you can filter for failures in Phoenix or Langfuse and see what went wrong.
"""

import sys

from time_grader import states_correct_times

from ai_playground.datasets import record_experiment, sync_dataset
from ai_playground.evals import Case, Experiment, pass_rate, print_report, run_evaluation
from ai_playground.llm import call_llm
from ai_playground.scores import publish_scores
from ai_playground.tools import get_current_time, get_current_time_in_timezone
from ai_playground.tracing import setup_tracing, traced_run

# The questions, and the timezones whose current time each answer must state.
# "local" means this computer's own timezone.
CASES = [
    Case("utc", "What time is it in UTC?", {"timezones": ["UTC"]}),
    Case("tokyo", "What time is it in Tokyo right now?", {"timezones": ["Asia/Tokyo"]}),
    Case("local", "What's the time here?", {"timezones": ["local"]}),
    Case(
        "local and utc",
        "What is the current local time and the current time in UTC?",
        {"timezones": ["local", "UTC"]},
    ),
    Case(
        "two cities",
        "What time is it in New York and in London?",
        {"timezones": ["America/New_York", "Europe/London"]},
    ),
    Case(
        "12-hour clock",
        "What time is it in Sydney? Use a 12-hour clock with AM or PM.",
        {"timezones": ["Australia/Sydney"]},
    ),
]

# The run counts as failed below this, so a script or CI job can check it.
MINIMUM_PASS_RATE = 0.8


def answer(question: str) -> str:
    """The application under test: the agent from experiment 01."""
    response = call_llm(question, tools=[get_current_time, get_current_time_in_timezone])
    return response.output_text


def main() -> None:
    setup_tracing()

    # Keep the cases as a dataset in Phoenix and Langfuse, so every run of
    # them is listed there and two runs can be compared case by case.
    dataset = sync_dataset("time-questions", CASES)

    # One session per run, so its six traces show up together, labelled with
    # the model and git commit. The run is also an experiment on the dataset.
    with traced_run("02_evaluation", tags=["eval"]) as run:
        experiment = Experiment(run.session_id, dataset, run.metadata)
        results = run_evaluation(CASES, answer, states_correct_times, experiment)
    print_report(results)
    print(f"Session and experiment: {run.session_id}")

    publish_scores(results)  # each verdict on its trace, in both tools
    record_experiment(experiment, results)  # the run on the dataset, in Phoenix

    if pass_rate(results) < MINIMUM_PASS_RATE:
        print(f"Below the minimum pass rate of {MINIMUM_PASS_RATE:.0%}.")
        sys.exit(1)


if __name__ == "__main__":
    main()
