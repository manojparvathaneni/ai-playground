"""Tests for the evaluation runner (ai_playground/evals.py)."""

from ai_playground.evals import Case, Grade, pass_rate, run_evaluation


def contains_expected_word(case: Case, answer: str) -> Grade:
    word = case.expected["word"]
    if word in answer:
        return Grade(True, f"mentions {word}")
    return Grade(False, f"doesn't mention {word}")


def test_each_case_is_answered_and_graded():
    cases = [
        Case("hello", "Say hello", {"word": "hello"}),
        Case("bye", "Say bye", {"word": "bye"}),
    ]

    results = run_evaluation(cases, lambda question: "hello there", contains_expected_word)

    assert [(r.case.name, r.answer, r.grade.passed) for r in results] == [
        ("hello", "hello there", True),
        ("bye", "hello there", False),
    ]
    assert pass_rate(results) == 0.5


def test_a_case_that_raises_fails_without_stopping_the_run():
    def flaky(question: str) -> str:
        if question == "break":
            raise RuntimeError("model server down")
        return "hello"

    cases = [Case("broken", "break", {"word": "hello"}), Case("fine", "ok", {"word": "hello"})]

    results = run_evaluation(cases, flaky, contains_expected_word)

    assert [r.grade.passed for r in results] == [False, True]
    assert "RuntimeError: model server down" in results[0].grade.explanation


def test_cases_are_traced_with_their_grade(span_exporter):
    run_evaluation(
        [Case("hello", "Say hello", {"word": "hello"})], lambda q: "hello", contains_expected_word
    )

    by_name = {s.name: s for s in span_exporter.get_finished_spans()}
    assert by_name["eval: hello"].attributes["openinference.span.kind"] == "CHAIN"
    grade = by_name["grade"]
    assert grade.attributes["openinference.span.kind"] == "EVALUATOR"
    assert grade.attributes["eval.passed"] is True
    assert grade.parent.span_id == by_name["eval: hello"].context.span_id


def test_pass_rate_of_no_results_is_zero():
    assert pass_rate([]) == 0.0
