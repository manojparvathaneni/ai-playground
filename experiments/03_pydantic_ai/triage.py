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

from dataclasses import dataclass
from typing import Literal, Self

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator
from pydantic_ai import Agent, ModelRetry, NativeOutput, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
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
    ]
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
    needs_escalation: bool = Field(description="Must be true for severity A.")
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
    def severity_matches_impact(self) -> Self:
        if self.severity == "A" and self.customer_impact in ("single_user", "question"):
            raise ValueError(f"severity A conflicts with customer_impact '{self.customer_impact}'")
        if self.customer_impact == "outage" and self.severity == "C":
            raise ValueError("an outage cannot be severity C")
        if self.severity == "A" and not self.needs_escalation:
            raise ValueError("severity A cases must set needs_escalation to true")
        return self

    @model_validator(mode="after")
    def low_confidence_needs_questions(self) -> Self:
        if self.confidence < 0.6 and not self.missing_info:
            raise ValueError(
                f"confidence is {self.confidence} but missing_info is empty; "
                "list what you need to know"
            )
        return self


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


def make_model(settings: Settings | None = None) -> OpenAIChatModel:
    """The model from Settings, through any OpenAI-compatible server (LM Studio by default)."""
    if settings is None:
        settings = Settings()
    provider = OpenAIProvider(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    return OpenAIChatModel(settings.llm_model, provider=provider)


# ---------------------------------------------------------------------------
# Cases to try
# ---------------------------------------------------------------------------

CASES = [
    (
        "Since 09:40 UTC all 4,000 users in our tenant get AADSTS50076 when signing in to "
        "Outlook and Teams. We changed a Conditional Access policy last night. Production is down.",
        "unified",
    ),
    (
        "One of our accountants cannot map the finance share. Windows shows error 0x80070035. "
        "Everyone else in the office is fine.",
        "standard",
    ),
    (
        "Our database feels slow sometimes. Not sure when it started. Can you take a look?",
        "standard",
    ),
    ("THIS IS URGENT SEV A!!! How do I export last month's invoices to CSV?", "unified"),
]
