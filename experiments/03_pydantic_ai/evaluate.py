"""Measure how often the triage agent is right, not just valid.

main.py shows that every case ends with a valid triage. This compares each
triage with the expected one (see triage_grader.py), using the same
evaluation tools as experiment 02: each run is a session, the cases are the
"triage-cases" dataset in Phoenix and Langfuse, and each run is an
experiment on it.
"""

import argparse
import os
import sys

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from triage import CASES, EXAMPLES, CaseDeps, make_agent, make_model  # noqa: E402
from triage_grader import triage_matches  # noqa: E402

from ai_playground.datasets import record_experiment, sync_dataset  # noqa: E402
from ai_playground.evals import Case, Experiment, pass_rate, print_report, run_evaluation  # noqa: E402
from ai_playground.scores import publish_scores  # noqa: E402
from ai_playground.tracing import setup_tracing, traced_run  # noqa: E402

# What a support engineer would decide for each case. Each field lists every
# acceptable answer; fields that aren't listed aren't checked.
EXPECTED = {
    "tenant outage": {
        "severity": ["A"],
        "product_area": ["identity"],
        "affected_users": [4000],
        "needs_escalation": [True],
    },
    "one user's share": {
        "severity": ["C"],  # one user, everyone else is fine
        "product_area": ["networking", "storage"],
        "affected_users": [1, None],
        "needs_escalation": [False],
    },
    "slow database": {
        "severity": ["B", "C"],  # too vague to tell
        "product_area": ["database"],
        "affected_users": [None],  # the case gives no number
        "asks_questions": True,
    },
    "invoice question": {
        "severity": ["C"],  # a how-to question, whatever the customer's label
        "product_area": ["billing"],
        "affected_users": [None],
        "needs_escalation": [False],
    },
}

# A case that no fix was tuned for, to check that the fixes don't break the
# cases they weren't aimed at: a real group of users is severity B.
HELD_OUT = {
    "branch VPN": (
        "About 60 people in our London office cannot connect to the VPN since this morning. "
        "Staff in other offices are fine.",
        "standard",
        {"severity": ["B"], "product_area": ["networking"], "affected_users": [60]},
    ),
}

CASES_TO_GRADE = [
    Case(name, text, {"plan": plan, **EXPECTED[name]}) for name, (text, plan) in CASES.items()
] + [
    Case(name, text, {"plan": plan, **expected})
    for name, (text, plan, expected) in HELD_OUT.items()
]

# The run counts as failed below this, as in experiment 02.
MINIMUM_PASS_RATE = 0.8


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        choices=["native", "tool"],
        default="native",
        help="how the schema reaches the model: as the response format, or as a tool to call",
    )
    parser.add_argument(
        "--examples", action="store_true", help="add the worked examples to the instructions"
    )
    args = parser.parse_args()

    setup_tracing(pydantic_ai=True)
    model = make_model()
    agent = make_agent(args.output)
    instructions = EXAMPLES if args.examples else None
    # The application under test only gets the question, so look up each
    # case's support plan by its text.
    plans = {case.question: case.expected["plan"] for case in CASES_TO_GRADE}

    dataset = sync_dataset("triage-cases", CASES_TO_GRADE)
    tags = ["eval", f"output-{args.output}"] + (["examples"] if args.examples else [])
    with traced_run("03_pydantic_ai_eval", tags=tags) as run:

        def triage(case_text: str) -> str:
            result = agent.run_sync(
                case_text,
                deps=CaseDeps(case_text, plans[case_text]),
                model=model,
                instructions=instructions,  # added to the agent's own
                conversation_id=run.session_id,  # keep the run's session (see main.py)
            )
            return result.output.model_dump_json()

        # The experiment records how this run was set up, so runs can be
        # compared in Phoenix and Langfuse.
        setup = {"output": args.output, "examples": str(args.examples).lower()}
        experiment = Experiment(run.session_id, dataset, {**run.metadata, **setup})
        results = run_evaluation(CASES_TO_GRADE, triage, triage_matches, experiment)

    print_report(results)
    print(f"Session and experiment: {run.session_id} ({setup})")
    publish_scores(results)
    record_experiment(experiment, results)

    if pass_rate(results) < MINIMUM_PASS_RATE:
        print(f"Below the minimum pass rate of {MINIMUM_PASS_RATE:.0%}.")
        sys.exit(1)


if __name__ == "__main__":
    main()
