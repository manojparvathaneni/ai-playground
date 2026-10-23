"""Run an AI application on a set of test questions and score the answers.

An evaluation ("eval") is to an AI application what tests are to ordinary
code. You can't assert one exact answer, because models phrase things
differently each time, so instead you write a grader: a function that
decides whether an answer is good enough. Run the same cases after every
change (a new prompt, a new model, a new tool) to see whether things got
better or worse.

Each case is traced, so you can open a failing case in Phoenix or Langfuse
and see exactly what happened:

  eval: <case name>          <- one case
  ├── agent                  <- the application answering the question
  │   └── ...
  └── grade                  <- the grader's verdict and explanation

Given an Experiment, the run is also recorded as an experiment on a dataset
in Phoenix and Langfuse, so runs can be compared case by case (datasets.py).
"""

import json
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from opentelemetry.trace import Span

from ai_playground.tracing import (
    SpanKind,
    experiment_context,
    experiment_item,
    set_output,
    span,
)


@dataclass
class Case:
    """One test question, plus whatever the grader needs to judge the answer."""

    name: str
    question: str
    expected: dict[str, Any] = field(default_factory=dict)


@dataclass
class Grade:
    """A grader's verdict on one answer."""

    passed: bool
    explanation: str


@dataclass
class Result:
    case: Case
    answer: str
    grade: Grade
    # The IDs of this case's trace and of its top span, so scores can be
    # attached to them (see scores.py). None if tracing isn't set up.
    trace_id: str | None = None
    span_id: str | None = None
    # When the case started and finished, for experiment runs (datasets.py).
    started_at: datetime | None = None
    ended_at: datetime | None = None


@dataclass
class Dataset:
    """A named set of cases, kept in Phoenix and Langfuse (see datasets.py)."""

    name: str
    cases: list[Case]
    # The IDs the two tools gave the dataset. None if it couldn't be created.
    phoenix_id: str | None = None
    langfuse_id: str | None = None

    def item_id(self, case: Case) -> str:
        """A case's ID in the dataset: the same in both tools and in every run."""
        return f"{self.name}-{case.name}".replace(" ", "-")


@dataclass
class Experiment:
    """One run of an evaluation on a dataset, usually named after its session."""

    name: str
    dataset: Dataset
    # Shown with the experiment, for example the model and git commit.
    metadata: dict[str, str] = field(default_factory=dict)


# The application being evaluated: takes a question, returns the answer text.
Application = Callable[[str], str]
# A grader: takes the case and the answer, returns a Grade.
Grader = Callable[[Case, str], Grade]


def run_evaluation(
    cases: list[Case],
    application: Application,
    grader: Grader,
    experiment: Experiment | None = None,
) -> list[Result]:
    """Answer and grade every case. A case that raises an error counts as failed.

    With an experiment, the traces are also labelled as that experiment's runs
    on its dataset, which is how Langfuse records experiments.
    """
    langfuse_dataset_id = experiment.dataset.langfuse_id if experiment else None
    if experiment and langfuse_dataset_id:
        labels = experiment_context(experiment.name, langfuse_dataset_id)
    else:
        labels = nullcontext()
    with labels:
        return [run_case(case, application, grader, experiment) for case in cases]


def run_case(
    case: Case, application: Application, grader: Grader, experiment: Experiment | None
) -> Result:
    """Answer and grade one case, in a trace of its own."""
    started_at = datetime.now(UTC)
    with span(f"eval: {case.name}", SpanKind.CHAIN, case.question) as case_span:
        if experiment and experiment.dataset.langfuse_id:
            item = experiment_item(
                case_span, experiment.dataset.item_id(case), json.dumps(case.expected)
            )
        else:
            item = nullcontext()
        with item:
            try:
                answer = application(case.question)
                grade = None
            except Exception as error:  # one broken case shouldn't stop the whole run
                answer = ""
                grade = Grade(False, f"The application raised {type(error).__name__}: {error}")

            with span("grade", SpanKind.EVALUATOR, answer) as grade_span:
                if grade is None:
                    grade = grader(case, answer)
                grade_span.set_attribute("eval.passed", grade.passed)
                grade_span.set_attribute("eval.explanation", grade.explanation)
                set_output(grade_span, "pass" if grade.passed else f"fail: {grade.explanation}")

        set_output(case_span, answer)
        trace_id, span_id = ids_of(case_span)
    return Result(case, answer, grade, trace_id, span_id, started_at, datetime.now(UTC))


def ids_of(current: Span) -> tuple[str | None, str | None]:
    """The span's trace ID (32 hex characters) and its own ID (16), the form
    Phoenix and Langfuse show. (None, None) if tracing isn't set up."""
    context = current.get_span_context()
    if not context.is_valid:  # tracing isn't set up, so there is no real trace
        return None, None
    return format(context.trace_id, "032x"), format(context.span_id, "016x")


def pass_rate(results: list[Result]) -> float:
    """The fraction of cases that passed, from 0.0 to 1.0."""
    if not results:
        return 0.0
    return sum(result.grade.passed for result in results) / len(results)


def print_report(results: list[Result]) -> None:
    """Print one line per case, then the overall pass rate."""
    for result in results:
        mark = "PASS" if result.grade.passed else "FAIL"
        print(f"{mark}  {result.case.name}")
        if not result.grade.passed:
            print(f"      question: {result.case.question}")
            print(f"      answer:   {result.answer}")
            print(f"      why:      {result.grade.explanation}")
    passed = sum(result.grade.passed for result in results)
    print(f"\n{passed}/{len(results)} passed ({pass_rate(results):.0%})")
