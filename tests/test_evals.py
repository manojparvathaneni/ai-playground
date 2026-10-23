"""Tests for the evaluation runner (ai_playground/evals.py)."""

from ai_playground.evals import Case, Dataset, Experiment, Grade, pass_rate, run_evaluation


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
    assert [r.error for r in results] == ["RuntimeError: model server down", None]


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


def test_results_record_their_span_and_timing(span_exporter):
    (result,) = run_evaluation(
        [Case("hello", "Say hello", {"word": "hello"})], lambda q: "hello", contains_expected_word
    )

    (root,) = [s for s in span_exporter.get_finished_spans() if s.name == "eval: hello"]
    assert result.trace_id == format(root.context.trace_id, "032x")
    assert result.span_id == format(root.context.span_id, "016x")
    assert result.started_at is not None and result.ended_at is not None
    assert result.started_at <= result.ended_at


def test_dataset_item_ids_are_stable_and_have_no_spaces():
    dataset = Dataset("time-questions", [])

    assert dataset.item_id(Case("local and utc", "?")) == "time-questions-local-and-utc"


def test_an_experiment_labels_each_case_as_a_dataset_item(span_exporter):
    case = Case("hello", "Say hello", {"word": "hello"})
    dataset = Dataset("greetings", [case], langfuse_id="langfuse-dataset-id")

    run_evaluation([case], lambda q: "hello", contains_expected_word, Experiment("run-1", dataset))

    for finished in span_exporter.get_finished_spans():  # the case and its grade
        assert finished.attributes["langfuse.experiment.name"] == "run-1"
        assert finished.attributes["langfuse.experiment.dataset.id"] == "langfuse-dataset-id"
        assert finished.attributes["langfuse.experiment.item.id"] == "greetings-hello"


def test_without_a_langfuse_dataset_nothing_is_labelled(span_exporter):
    case = Case("hello", "Say hello", {"word": "hello"})
    dataset = Dataset("greetings", [case])  # Langfuse couldn't be reached

    run_evaluation([case], lambda q: "hello", contains_expected_word, Experiment("run-1", dataset))

    for finished in span_exporter.get_finished_spans():
        assert not any(key.startswith("langfuse.") for key in finished.attributes)
