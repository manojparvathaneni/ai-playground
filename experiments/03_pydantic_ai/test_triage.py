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
from pydantic_ai.messages import ThinkingPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from triage import CaseDeps, TriageValidated, make_model, triage_agent

from ai_playground.config import Settings

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


def test_a_count_in_the_case_is_filled_in_when_missing():
    assert validate({**GOOD, "affected_users": None}).affected_users == 4000


def test_a_number_written_as_a_word_counts():
    text = "One of our accountants cannot map the finance share."
    one_user = {
        **GOOD,
        "severity": "C",
        "customer_impact": "single_user",
        "affected_users": 1,
        "error_codes": [],
        "needs_escalation": False,
    }

    assert TriageValidated.model_validate(one_user, context={"source_text": text})


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"error_codes": ["AADSTS50076", "AADSTS700016"]}, "not found in the case text"),
        ({"customer_impact": "question"}, "severity A and customer_impact 'question' don't fit"),
        ({"customer_impact": "question", "severity": "B"}, "Fix whichever is wrong"),
        (
            {"customer_impact": "question", "severity": "C"},
            "severity C cases must set needs_escalation to false",
        ),
        ({"affected_users": 400}, "never states that number"),
        (
            {"customer_impact": "single_user", "severity": "C", "needs_escalation": False},
            "the case says 4000 users are affected",
        ),
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
    assert "severity A and customer_impact 'question' don't fit together" in prompt


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


def test_an_answer_sent_as_thinking_is_never_accepted():
    # What LM Studio does with a thinking model and JSON output: the answer
    # arrives as reasoning, and the reply itself is empty. Pydantic AI asks
    # for text each time, the model does the same thing again, and the run
    # gives up. Retries can't fix this; LLM_REASONING_EFFORT=none does.
    def thinks_only(messages, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ThinkingPart(json.dumps(RIGHT))])

    with pytest.raises(UnexpectedModelBehavior, match="Exceeded maximum output retries"):
        run(INVOICE_CASE, "standard", FunctionModel(thinks_only))


def test_reasoning_effort_is_sent_only_when_set():
    unset = Settings(_env_file=None, llm_reasoning_effort=None)
    none = Settings(_env_file=None, llm_reasoning_effort="none")

    assert "openai_reasoning_effort" not in (make_model(unset).settings or {})
    assert (make_model(none).settings or {})["openai_reasoning_effort"] == "none"


def test_a_group_labelled_single_user_is_told_the_impact_is_wrong():
    # Qwen's answer for 60 people who can't reach the VPN. The severity is
    # fine; the impact isn't, and the feedback has to say so.
    text = "About 60 people in our London office cannot connect to the VPN since this morning."
    answer = {**GOOD, "customer_impact": "single_user", "affected_users": None, "error_codes": []}

    with pytest.raises(ValidationError, match="the case says 60 users are affected") as raised:
        TriageValidated.model_validate(answer, context={"source_text": text})
    assert "fit together" not in str(raised.value)  # not sent to change the severity
