# 03 · Structured output with Pydantic AI

A model triages customer support cases into a record an engineer can act
on: product area, severity A/B/C, impact, summary, error codes, next steps,
escalation, confidence and open questions. [Pydantic](https://pydantic.dev)
checks every reply, and a [Pydantic AI](https://ai.pydantic.dev) agent sends
failed checks back to the model until it gets a valid one.

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

It prints each case's triage, how many requests it took, a tally and the
session ID. In Phoenix or Langfuse, open the session to see the four runs;
for a retried case, open the second `chat` span to read the validation
errors Pydantic AI sent back.

## What it found

In the first runs with `google/gemma-4-e4b`, all four cases ended valid:
three on the first try, and the invoice question ("THIS IS URGENT SEV A!!!
How do I export last month's invoices to CSV?") after one retry.

Valid isn't the same as right, though. The validators catch broken rules,
not poor judgement, and these passed them:

- The accountant who can't map a share was filed under `identity`, severity
  B. It's a single user with a networking or storage error, so C fits better.
- The vague "database feels slow" case got `affected_users: 1`, a number the
  case never states.
- The invoice question was severity B, not C: the customer's "SEV A" label
  still pulled it up a level.

Catching these takes expected answers to compare against, which is what an
evaluation does (see [experiment 02](../02_evaluation/)).

## Try next

- **Turn it into an evaluation.** Give each case an expected severity and
  product area, grade the triage against them with `run_evaluation()`, and
  compare models on the dataset in Phoenix or Langfuse.
- **Add a rule.** Make `affected_users` fail when the case text has no
  number in it, and see whether the retry fixes the "database" case.
- **Change the output mode.** Swap `NativeOutput` for `ToolOutput` (the
  schema becomes a tool the model must call) or `PromptedOutput` (the schema
  goes in the prompt), and compare how many retries each needs.
- **Drop the severity line** from `SYSTEM_PROMPT` and watch the invoice case,
  where the customer's own label says "SEV A".
