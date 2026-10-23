"""Offline tests for the triage validators and agent. No model is called:
FunctionModel stands in for it, and real model requests are switched off."""

import json

import pytest
from pydantic import ValidationError
from pydantic_ai import (
    ModelResponse,
    TextPart,
    UnexpectedModelBehavior,
    capture_run_messages,
    models,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from triage import CaseDeps, TriageValidated, triage_agent

# Any request to a real model fails the test instead of being sent.
models.ALLOW_MODEL_REQUESTS = False

CASE = (
    "Since 09:40 UTC all 4,000 users get AADSTS50076 when signing in to Teams. Production is down."
)

GOOD = {
    "product_area": "identity",
    "severity": "A",
    "customer_impact": "outage",
    "issue_summary": "Tenant wide sign in failures with AADSTS50076 since 09:40 UTC.",
    "affected_users": 4000,
    "error_codes": ["AADSTS50076"],
    "next_steps": ["Review Conditional Access changes from the last 24 hours"],
    "needs_escalation": True,
    "confidence": 0.9,
    "missing_info": [],
}


def validate(data):
    return TriageValidated.model_validate(data, context={"source_text": CASE})


def test_a_good_triage_passes():
    assert validate(GOOD).severity == "A"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"error_codes": ["AADSTS50076", "AADSTS700016"]}, "not found in the case text"),
        ({"customer_impact": "single_user"}, "severity A conflicts with customer_impact"),
        ({"needs_escalation": False}, "must set needs_escalation to true"),
        ({"confidence": 0.4}, "missing_info is empty"),
        ({"severity": "P1"}, "Input should be 'A', 'B' or 'C'"),
    ],
)
def test_each_rule_catches_its_mistake(change, message):
    with pytest.raises(ValidationError, match=message):
        validate({**GOOD, **change})


INVOICE_CASE = "THIS IS URGENT SEV A!!! How do I export last month's invoices to CSV?"
WRONG = {
    "product_area": "billing",
    "severity": "A",  # a how-to question can't be severity A
    "customer_impact": "question",
    "issue_summary": "Customer asks how to export last month's invoices to CSV.",
    "affected_users": None,
    "error_codes": [],
    "next_steps": ["Send the invoice export steps"],
    "needs_escalation": True,
    "confidence": 0.9,
    "missing_info": [],
}
RIGHT = {**WRONG, "severity": "C", "needs_escalation": False}


def answers(*replies):
    """A fake model that gives these replies in turn, one per request."""

    def model(messages, info: AgentInfo) -> ModelResponse:
        requests = sum(1 for message in messages if message.kind == "request")
        return ModelResponse(parts=[TextPart(json.dumps(replies[requests - 1]))])

    return FunctionModel(model)


def run(case_text, plan, model):
    with capture_run_messages() as messages, triage_agent.override(model=model):
        result = triage_agent.run_sync(case_text, deps=CaseDeps(case_text, plan))
    return result.output, messages


def retry_prompts(messages):
    return [
        part.model_response()
        for message in messages
        for part in message.parts
        if part.part_kind == "retry-prompt"
    ]


def test_a_failed_check_is_sent_back_and_the_model_corrects_itself():
    output, messages = run(INVOICE_CASE, "standard", answers(WRONG, RIGHT))

    assert output.severity == "C"
    (prompt,) = retry_prompts(messages)
    assert "severity A conflicts with customer_impact 'question'" in prompt


def test_the_unified_plan_rule_uses_the_deps():
    severity_b = {**RIGHT, "severity": "B", "customer_impact": "degraded"}
    escalated = {**severity_b, "needs_escalation": True}

    output, messages = run(INVOICE_CASE, "unified", answers(severity_b, escalated))

    assert output.needs_escalation
    (prompt,) = retry_prompts(messages)
    assert "Unified plan" in prompt


def test_the_agent_gives_up_after_two_retries():
    with pytest.raises(UnexpectedModelBehavior, match="Exceeded maximum output retries"):
        run(INVOICE_CASE, "standard", answers(WRONG, WRONG, WRONG))
