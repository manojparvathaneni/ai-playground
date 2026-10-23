"""Tests for the time grader. A grader with a bug gives misleading scores."""

from unittest.mock import patch

import pytest
from time_grader import close_enough, states_correct_times, times_in

from ai_playground.evals import Case


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("It is 16:05.", [16 * 60 + 5]),
        ("16:05:09", [16 * 60 + 5]),
        ("4:05 PM", [16 * 60 + 5]),
        ("4:05 p.m.", [16 * 60 + 5]),
        ("9:10am", [9 * 60 + 10]),
        ("12:06 AM", [6]),  # just after midnight
        ("12:06 PM", [12 * 60 + 6]),  # just after noon
        ("16:05 and 9:10", [16 * 60 + 5, 9 * 60 + 10]),  # the "a" of "and" isn't AM
        ("4:05 PM in Sydney, and 11:05 AM in Tokyo", [16 * 60 + 5, 11 * 60 + 5]),
        ("no times here", []),
    ],
)
def test_times_in(text, expected):
    assert times_in(text) == expected


def test_close_enough_across_midnight():
    assert close_enough(23 * 60 + 59, 0)
    assert not close_enough(0, 3)


def test_wrong_am_pm_fails():
    # Six minutes past midnight, but the model said "PM": the mistake from
    # experiment 01 that this grader is meant to catch.
    case = Case("local", "What's the time here?", {"timezones": ["local"]})
    with patch("time_grader.now_in", return_value=6):
        assert states_correct_times(case, "It's 12:06 AM.").passed
        grade = states_correct_times(case, "It's 12:06 PM.")
    assert not grade.passed
    assert "local should be about 00:06" in grade.explanation
