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
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from opentelemetry.trace import Span

from ai_playground.tracing import SpanKind, set_output, span


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
    # The ID of this case's trace, so scores can be attached to it (see
    # scores.py). None if tracing isn't set up.
    trace_id: str | None = None


# The application being evaluated: takes a question, returns the answer text.
Application = Callable[[str], str]
# A grader: takes the case and the answer, returns a Grade.
Grader = Callable[[Case, str], Grade]


def run_evaluation(cases: list[Case], application: Application, grader: Grader) -> list[Result]:
    """Answer and grade every case. A case that raises an error counts as failed."""
    results = []
    for case in cases:
        with span(f"eval: {case.name}", SpanKind.CHAIN, case.question) as case_span:
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
            trace_id = trace_id_of(case_span)
        results.append(Result(case, answer, grade, trace_id))
    return results


def trace_id_of(current: Span) -> str | None:
    """The span's trace ID as 32 hex characters, the form Phoenix and Langfuse show."""
    context = current.get_span_context()
    if not context.is_valid:  # tracing isn't set up, so there is no real trace
        return None
    return format(context.trace_id, "032x")


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
