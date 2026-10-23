"""Grade a triage against what a support engineer would have decided.

The validators in triage.py only check that a triage is consistent and
grounded; a valid triage can still be wrong. This grader compares it with
the expected answer for each case: the severity, the product area, how many
users the case says are affected, whether to escalate, and whether a vague
case asks questions.

Each expected value lists every answer that's acceptable, because some cases
honestly allow more than one (a share that won't map can be filed under
networking or storage).
"""

import json

from ai_playground.evals import Case, Grade


def triage_matches(case: Case, answer: str) -> Grade:
    """Pass if every expected field of the triage is one of the accepted values."""
    try:
        triage = json.loads(answer)
    except json.JSONDecodeError:
        return Grade(False, "no valid triage (the agent gave up or failed)")

    expected = case.expected
    wrong = []
    for field in ("severity", "product_area", "affected_users", "needs_escalation"):
        if field in expected and triage.get(field) not in expected[field]:
            accepted = " or ".join(repr(value) for value in expected[field])
            wrong.append(f"{field} is {triage.get(field)!r}, expected {accepted}")
    if expected.get("asks_questions") and not triage.get("missing_info"):
        wrong.append("the case is vague, but missing_info is empty")

    if wrong:
        return Grade(False, "; ".join(wrong))
    return Grade(True, "matches the expected triage")
