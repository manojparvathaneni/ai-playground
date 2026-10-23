"""A support case triage agent: Pydantic models checked by Pydantic AI.

The model reads a customer's support case and returns a triage record an
engineer can act on. Two classes split the job, matching two layers of
checking:

  TriageSchema     the shape: field names, types, allowed values, lengths
                   and ranges. It's sent to the model server as the response
                   format, so the server can enforce it while generating.
  TriageValidated  the meaning: rules across fields (severity must match
                   impact) and against the case text (no invented error
                   codes). These can't be written in JSON Schema, so they
                   run in Python after the reply arrives.

triage_agent runs the loop: ask the model, validate the reply, and when a
check fails, send the errors back and ask again, up to two more times.
"""

import re
from dataclasses import dataclass
from typing import Literal, Self

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator
from pydantic_ai import Agent, ModelRetry, NativeOutput, RunContext
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

from ai_playground.config import Settings

# ---------------------------------------------------------------------------
# The output: shape and meaning
# ---------------------------------------------------------------------------


class TriageSchema(BaseModel):
    """The shape. Sent to the model server, so it shapes what the model writes.

    Every description below is part of the prompt: the model reads them.
    """

    product_area: Literal[
        "identity", "networking", "storage", "compute", "database", "billing", "other"
    ] = Field(
        description="identity: sign-in, accounts, MFA, permissions. "
        "networking: VPN, DNS, connectivity, and network paths or shares that can't be "
        "reached. storage: disks, file and blob data, capacity, backups. "
        "compute: virtual machines and containers. database: database servers and queries. "
        "billing: invoices, payments, subscriptions."
    )
    severity: Literal["A", "B", "C"] = Field(
        description="A: production down or broad business impact. B: degraded service or a "
        "meaningful group affected. C: single user, how-to question, or advisory."
    )
    customer_impact: Literal["outage", "degraded", "single_user", "question"]
    issue_summary: str = Field(
        min_length=20, max_length=300, description="One or two plain sentences."
    )
    affected_users: int | None = Field(
        default=None, ge=1, description="Only if the case states a number, otherwise null."
    )
    error_codes: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Error codes copied exactly as written in the case. Never invent one. "
        "Empty list if none.",
    )
    next_steps: list[str] = Field(
        min_length=1, max_length=5, description="Concrete actions for the engineer."
    )
    needs_escalation: bool = Field(
        description="True for severity A. False for severity C. For severity B, true only "
        "when the customer's plan calls for it."
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="How sure you are of severity and product_area."
    )
    missing_info: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Questions to ask the customer. Required when confidence is below 0.6.",
    )


class TriageValidated(TriageSchema):
    """The meaning: rules across fields, and checks against the case text."""

    @field_validator("error_codes")
    @classmethod
    def codes_must_appear_in_case(cls, codes: list[str], info: ValidationInfo) -> list[str]:
        # The case text arrives as "validation context", not in the reply.
        source = (info.context or {}).get("source_text")
        if source is None:
            return codes
        invented = [code for code in codes if code.lower() not in source.lower()]
        if invented:
            raise ValueError(f"error codes not found in the case text: {invented}")
        return codes

    @model_validator(mode="after")
    def affected_users_matches_case(self, info: ValidationInfo) -> Self:
        # Runs before severity_matches_impact(): Pydantic runs model
        # validators in the order they're written and stops at the first that
        # fails, and this one can say precisely when the impact is the wrong
        # field. Feedback that names the wrong field is feedback the model
        # can't act on.
        source = (info.context or {}).get("source_text")
        if source is None:
            return self
        stated = COUNTED_USERS.search(source)
        if self.affected_users is None and stated:
            # A fact the code can read from the case: fill it in rather than
            # asking the model again. Validators can repair, not just reject.
            self.affected_users = int(stated.group(1).replace(",", ""))
        elif self.affected_users is not None and self.affected_users not in numbers_in(source):
            raise ValueError(
                f"affected_users is {self.affected_users}, but the case never states that "
                "number; use null unless the case gives one"
            )
        if self.customer_impact == "single_user" and (self.affected_users or 1) > 1:
            raise ValueError(
                f"customer_impact is 'single_user', but the case says {self.affected_users} "
                "users are affected; choose 'outage' or 'degraded' instead"
            )
        return self

    @model_validator(mode="after")
    def severity_matches_impact(self) -> Self:
        # The severity definitions above, as rules.
        if self.customer_impact in ("single_user", "question") and self.severity != "C":
            raise ValueError(
                f"severity {self.severity} and customer_impact '{self.customer_impact}' don't "
                "fit together: C covers a single user or a how-to question, whatever label "
                "the customer used. Fix whichever is wrong: the severity, or the impact if "
                "more than one user is affected"
            )
        if self.customer_impact == "outage" and self.severity == "C":
            raise ValueError("an outage cannot be severity C")
        if self.severity == "A" and not self.needs_escalation:
            raise ValueError("severity A cases must set needs_escalation to true")
        if self.severity == "C" and self.needs_escalation:
            raise ValueError("severity C cases must set needs_escalation to false")
        return self

    @model_validator(mode="after")
    def low_confidence_needs_questions(self) -> Self:
        if self.confidence < 0.6 and not self.missing_info:
            raise ValueError(
                f"confidence is {self.confidence} but missing_info is empty; "
                "list what you need to know"
            )
        return self


# Numbers written as words, for cases like "one of our accountants".
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "ten": 10}
# A count of people, such as "4,000 users" or "60 people".
COUNTED_USERS = re.compile(r"\b(\d[\d,]*) (?:users|people|staff|employees)\b", re.IGNORECASE)


def numbers_in(text: str) -> set[int]:
    """Every number written in the text, in digits ("4,000") or as a word ("one")."""
    digits = {int(found.replace(",", "")) for found in re.findall(r"\d[\d,]*", text)}
    words = {
        value
        for word, value in NUMBER_WORDS.items()
        if re.search(rf"\b{word}\b", text, re.IGNORECASE)
    }
    return digits | words


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You triage technical support cases for a cloud platform. Read the case and fill every field. "
    "Judge severity from the business impact described, not from the customer's own label. "
    "Severity A always needs escalation. "
    "Copy error codes exactly as written and never invent them. "
    "If the case is vague, lower your confidence and list the questions you would ask."
)


@dataclass
class CaseDeps:
    """What the agent knows besides the reply: its "dependencies".

    Validators and tools read these through RunContext. The support plan
    stands for data from another system, such as a CRM: it isn't in the case
    text, but some rules depend on it.
    """

    case_text: str
    support_plan: Literal["unified", "standard"]


triage_agent = Agent(
    deps_type=CaseDeps,
    # NativeOutput sends the schema as the server's JSON response format, the
    # way LM Studio can enforce it while generating.
    output_type=NativeOutput(TriageValidated),
    instructions=SYSTEM_PROMPT,
    # When validation fails, send the errors back and try again, twice at most.
    retries={"output": 2},
    # Lets codes_must_appear_in_case() see the case text.
    validation_context=lambda ctx: {"source_text": ctx.deps.case_text},
)


@triage_agent.output_validator
def unified_plan_escalates(ctx: RunContext[CaseDeps], output: TriageValidated) -> TriageValidated:
    """A rule that needs data from outside the reply: Unified plan customers
    get escalation for severity B as well as A."""
    unified = ctx.deps.support_plan == "unified"
    if unified and output.severity in ("A", "B") and not output.needs_escalation:
        raise ModelRetry(
            "This customer is on a Unified plan, so severity A or B cases must set "
            "needs_escalation to true."
        )
    return output


def make_model(settings: Settings | None = None) -> Model:
    """The model from Settings: Claude, or any OpenAI-compatible server (LM Studio by default)."""
    if settings is None:
        settings = Settings()
    if settings.llm_provider == "anthropic":
        return make_claude_model(settings)
    provider = OpenAIProvider(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    model_settings = OpenAIChatModelSettings()
    if settings.llm_reasoning_effort is not None:
        # See llm_reasoning_effort in config.py: without "none", a thinking
        # model's answer arrives as reasoning, the reply is empty, and every
        # retry gets "Please return text." and fails the same way.
        model_settings["openai_reasoning_effort"] = settings.llm_reasoning_effort
    return OpenAIChatModel(settings.llm_model, provider=provider, settings=model_settings)


def make_claude_model(settings: Settings) -> AnthropicModel:
    """Claude through Anthropic's own API, not its OpenAI-compatible endpoint.

    The OpenAI-compatible endpoint ignores response_format, which is how
    NativeOutput sends the schema, so Claude would only be asked for the
    shape, not held to it. Here NativeOutput becomes Claude's structured
    output, which enforces the schema while it writes, as LM Studio does.
    """
    effort = settings.llm_reasoning_effort or "medium"
    if effort not in ("low", "medium", "high"):
        raise ValueError(f"LLM_REASONING_EFFORT={effort} isn't a Claude effort level")
    provider = AnthropicProvider(api_key=settings.anthropic_api_key)
    # Thinking is on by default for current Claude models; effort sets how
    # much it thinks and writes. Set explicitly, rather than relying on the
    # default, so a change in the default doesn't change the results.
    return AnthropicModel(
        settings.llm_model,
        provider=provider,
        settings=AnthropicModelSettings(anthropic_effort=effort),
    )


# ---------------------------------------------------------------------------
# Cases to try
# ---------------------------------------------------------------------------

# Each case's name, its text, and the customer's support plan.
CASES = {
    "tenant outage": (
        "Since 09:40 UTC all 4,000 users in our tenant get AADSTS50076 when signing in to "
        "Outlook and Teams. We changed a Conditional Access policy last night. Production is down.",
        "unified",
    ),
    "one user's share": (
        "One of our accountants cannot map the finance share. Windows shows error 0x80070035. "
        "Everyone else in the office is fine.",
        "standard",
    ),
    "slow database": (
        "Our database feels slow sometimes. Not sure when it started. Can you take a look?",
        "standard",
    ),
    "invoice question": (
        "THIS IS URGENT SEV A!!! How do I export last month's invoices to CSV?",
        "unified",
    ),
}
