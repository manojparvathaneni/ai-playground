# 03 · Structured output with Pydantic AI

A model triages customer support cases into a record an engineer can act
on: product area, severity A/B/C, impact, summary, error codes, next steps,
escalation, confidence and open questions. [Pydantic](https://pydantic.dev)
checks every reply, and a [Pydantic AI](https://pydantic.dev/docs/ai/core-concepts/agent/)
agent sends failed checks back to the model until it gets a valid one.

New to Pydantic? Read [docs/pydantic.md](../../docs/pydantic.md) first: it
covers every idea used here, with examples and exercises.

## What it shows

- **Two layers of checking** ([`triage.py`](triage.py)):
  - `TriageSchema` is the **shape**: field names, types, allowed values,
    lengths, ranges. It goes to the model server as the response format, so
    LM Studio can enforce it while the model writes. Every `description` is
    part of the prompt.
  - `TriageValidated` adds the **meaning**: severity must match impact,
    severity A must escalate, low confidence must come with questions, and
    every error code must appear in the customer's own words. JSON Schema
    can't express these, so they run in Python after the reply arrives.
- **What the framework takes over.** Without a framework you'd write the
  loop yourself: call, validate, append the errors, call again. Pydantic AI
  does that:

  | By hand | Pydantic AI |
  |---|---|
  | `response_format=TriageSchema` | `output_type=NativeOutput(TriageValidated)` |
  | `model_validate(..., context={"source_text": ...})` | `validation_context=lambda ctx: {...}` |
  | append the errors and call again | automatic: failed checks go back as a retry prompt |
  | a `max_attempts` counter | `retries={"output": 2}` |
  | passing the case text around | `deps`, typed by `deps_type`, read through `RunContext` |
  | rules that need outside data | `@triage_agent.output_validator` raising `ModelRetry` |

  The models and validators are yours either way: they're the part that
  matters, and they'd move unchanged to another framework.
- **Rules that need outside data.** The support plan stands for data from a
  CRM: it's not in the case text. `unified_plan_escalates()` reads it from the
  deps and asks the model to retry if a Unified plan case at severity B isn't
  escalated.
- **Testing an agent offline** ([`test_triage.py`](test_triage.py)):
  `FunctionModel` replaces the model with a Python function, `override()`
  swaps it in, and `ALLOW_MODEL_REQUESTS = False` makes any real call fail.
  The tests check each validator, a retry that fixes the answer, and the
  agent giving up after two retries.
- **Valid isn't right** ([`evaluate.py`](evaluate.py),
  [`triage_grader.py`](triage_grader.py)): validators check that a triage is
  consistent, not that it's correct. The evaluation compares each triage
  with what a support engineer would decide, with the same tools as
  experiment 02: a session per run, the `triage-cases` dataset, and an
  experiment per run in Phoenix and Langfuse.
- **Repair, don't just reject.** A validator can fix a value as well as
  reject it. Facts the code can read from the case (such as "4,000 users")
  are filled in rather than sent back to the model.
- **Framework tracing.** `setup_tracing(pydantic_ai=True)` records Pydantic
  AI's own spans, so a retried case shows as one `invoke_agent` span with two
  `chat` calls under it (see
  [framework spans](../../docs/tracing.md#framework-spans-pydantic-ai)).

## Run it

Needs the `pydantic-ai` dependency group, which `make install` includes, and
a model server like the other experiments (LM Studio with
`google/gemma-4-e4b` by default; set `LLM_MODEL` and `LLM_BASE_URL` in
`.env` to use another).

```sh
uv run python experiments/03_pydantic_ai/main.py
```

To measure how often the triage is right, run the evaluation:

```sh
uv run python experiments/03_pydantic_ai/evaluate.py
```

`main.py` prints each case's triage, how many requests it took, a tally and
the session ID. In Phoenix or Langfuse, open the session to see the four runs;
for a retried case, open the second `chat` span to read the validation
errors Pydantic AI sent back.

## What it found

**Valid isn't right.** In the first runs, all four cases ended with a
valid triage, but several were wrong: the accountant's share was filed under
`identity` at severity B, the vague database case got a made-up
`affected_users: 1`, and the how-to question was severity B because the
customer wrote "SEV A". The validators only catch broken rules.

So [`evaluate.py`](evaluate.py) grades each triage against the expected
answer: the four cases, plus one held-out case (60 users who can't reach
the VPN, which really is severity B) to check that fixes don't break cases
they weren't aimed at. Each set of results is three runs with
`google/gemma-4-e4b`, because one run can pass or fail by luck:

| Version | Runs (of 5) | Average |
|---|---|---|
| The validators from the notebook | 2, 1, 3 | 40% |
| Stricter rules: severity C for a single user or a question, C never escalates, `affected_users` must match the case | 3, 2, 1 | 40% |
| Facts filled in by code, and feedback that names the wrong field | 3, 3, 2 | 53% |

What the middle row taught:

- **A small model often ignores feedback.** Replaying a failing case showed
  the model sending exactly the same answer after each retry. A rule it
  can't satisfy turns a wrong-but-valid answer into a run that gives up
  after its retries.
- **The message has to point at the field that's wrong.** "single_user must
  be severity C" made sense for a how-to question, but for a 4,000-user
  outage labelled `single_user`, the impact was wrong, not the severity.

So the last version stops asking the model for facts the code can read: if
the case says "4,000 users", `affected_users_matches_case()` fills it in,
and then a precise rule can say that `customer_impact` can't be
`single_user`. Invented numbers are still rejected, which the model can fix
by using null.

What's still wrong is judgement: the invoice question sometimes stays at A
(the model changes the impact instead of the severity), the vague database
case sometimes asks no questions, and the share is often filed under
`identity`. More rules won't fix judgement in a 4B model. The next step is
a bigger one: `qwen/qwen3.6-35b-a3b` is in LM Studio, but this machine
couldn't load it alongside Gemma (LM Studio stopped it to avoid running out
of memory). Unload Gemma first, then run
`LLM_MODEL=qwen/qwen3.6-35b-a3b uv run python experiments/03_pydantic_ai/evaluate.py`
and compare the experiments on `triage-cases`.

## Try next

- **Compare models.** Run the evaluation with another `LLM_MODEL` and compare
  the experiments on the `triage-cases` dataset, case by case.
- **Add a case.** A case that's hard for the current rules: a group outage
  described without a number, or a question that mentions an error code.
- **Change the output mode.** Swap `NativeOutput` for `ToolOutput` (the
  schema becomes a tool the model must call) or `PromptedOutput` (the schema
  goes in the prompt), and compare how many retries each needs.
- **Drop the severity line** from `SYSTEM_PROMPT` and watch the invoice case,
  where the customer's own label says "SEV A".
