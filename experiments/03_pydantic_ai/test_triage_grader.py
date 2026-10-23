"""Tests for the triage grader."""

import json

from triage_grader import triage_matches

from ai_playground.evals import Case

CASE = Case(
    "share",
    "One of our accountants cannot map the finance share.",
    {
        "severity": ["C"],
        "product_area": ["networking", "storage"],
        "affected_users": [1, None],
        "needs_escalation": [False],
    },
)
TRIAGE = {
    "severity": "C",
    "product_area": "storage",
    "affected_users": 1,
    "needs_escalation": False,
    "missing_info": [],
}


def grade(**changes):
    return triage_matches(CASE, json.dumps({**TRIAGE, **changes}))


def test_a_matching_triage_passes():
    assert grade().passed


def test_any_accepted_value_passes():
    assert grade(product_area="networking", affected_users=None).passed


def test_each_wrong_field_is_explained():
    result = grade(severity="B", product_area="identity")

    assert not result.passed
    assert "severity is 'B', expected 'C'" in result.explanation
    assert "product_area is 'identity', expected 'networking' or 'storage'" in result.explanation


def test_a_vague_case_must_ask_questions():
    vague = Case("slow", "It feels slow.", {"asks_questions": True})

    assert not triage_matches(vague, json.dumps(TRIAGE)).passed
    assert triage_matches(vague, json.dumps({**TRIAGE, "missing_info": ["When?"]})).passed


def test_no_triage_fails():
    assert not triage_matches(CASE, "").passed
