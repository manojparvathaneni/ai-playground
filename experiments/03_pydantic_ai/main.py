"""Experiment 03: structured output with Pydantic, through a Pydantic AI agent.

Triages four support cases. For each, the agent asks the model for a
TriageValidated record, checks it, and sends any failed checks back for
another try (see triage.py). The output shows how many requests each case
needed, so you can see the retries at work.
"""

import os

# Pydantic AI prints a welcome banner on import; keep the output clean.
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from pydantic_ai import capture_run_messages  # noqa: E402
from triage import CASES, CaseDeps, make_model, triage_agent  # noqa: E402

from ai_playground.tracing import setup_tracing, traced_run  # noqa: E402


def main():
    # pydantic_ai=True traces the agent with Pydantic AI's own spans.
    setup_tracing(pydantic_ai=True)
    model = make_model()  # the model and server from .env, LM Studio by default

    tally = {"first try": 0, "corrected": 0, "failed": 0}
    with traced_run("03_pydantic_ai") as run:
        for case_text, plan in CASES:
            print("=" * 80)
            print(f"CASE ({plan} plan): {case_text[:70]}...")
            # capture_run_messages() records every message sent and received,
            # including the retry prompts Pydantic AI writes for us.
            with capture_run_messages() as messages:
                try:
                    result = triage_agent.run_sync(
                        case_text,
                        deps=CaseDeps(case_text, plan),
                        model=model,
                        # Pydantic AI gives each run a new conversation ID,
                        # which the tools show as its session. Use the
                        # run's session instead, so the four cases share it.
                        conversation_id=run.session_id,
                    )
                except Exception as error:  # gave up after its retries
                    print(f"FAILED: {type(error).__name__}: {error}")
                    tally["failed"] += 1
                    continue

            requests = sum(1 for message in messages if message.kind == "request")
            print(f"VALID after {requests} request(s)")
            print(result.output.model_dump_json(indent=2))
            tally["first try" if requests == 1 else "corrected"] += 1

    print("=" * 80)
    print(f"Summary: {tally}")
    print(f"Session: {run.session_id}")


if __name__ == "__main__":
    main()
