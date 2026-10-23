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

To use Claude instead of LM Studio, add your key to `.env` and pick the
model there or on the command line:

```sh
LLM_PROVIDER=anthropic LLM_MODEL=claude-haiku-5-5 \
  uv run python experiments/03_pydantic_ai/evaluate.py
```

This goes through Anthropic's own API (`AnthropicModel` in `make_model()`),
not its OpenAI-compatible endpoint: that endpoint ignores `response_format`,
which is how `NativeOutput` sends the schema. Claude Haiku 5.5 costs $0.10
per million input tokens and $0.50 per million output tokens, so a run of
the evaluation costs a fraction of a cent.

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
`identity`. More rules won't fix judgement in a 4B model.

### A bigger model, and two retry problems

`qwen/qwen3.6-35b-a3b` is in LM Studio too (load it on its own: this
machine can't hold it and Gemma together). Its first runs failed in a new
way: the held-out VPN case gave up after its retries, and every retry prompt
said only "Please return text."

1. **The answer arrived as "thinking".** Qwen is a thinking model, and with
   JSON output LM Studio filed its whole answer as reasoning and sent an
   empty reply. Pydantic AI found no text, asked for some, and got the same
   empty reply each time: **retries can't fix a problem the model can't
   see.** The fix is a setting, not a rule: `LLM_REASONING_EFFORT=none`
   (see `config.py`) turns thinking off, and the answer comes back as text.
2. **The first failing check named the wrong field.** With the answer
   visible, Qwen labelled the 60 VPN users `single_user` at severity A.
   `severity_matches_impact()` ran first and said "single_user must be
   severity C", which is wrong for 60 users, so Qwen refused and sent the
   same answer until the retries ran out. The precise message ("the case
   says 60 users are affected; choose outage or degraded") was in a later
   validator that never ran, because Pydantic stops at the first model
   validator that fails. Running that validator first fixed it: Qwen
   corrects the impact on its first retry.

Three runs each with Qwen, thinking off:

| Version | Runs (of 5) | Average | Runs that gave up |
|---|---|---|---|
| Before the validator order fix | 4, 4, 3 | 73% | 2 |
| After | 4, 3, 4 | 73% | 0 |

The pass rate stayed the same, but every failure is now a judgement the
grader can score (Qwen rates the 60-user VPN case A, not B), rather than a
run that wasted its retries. Qwen also gets all four original cases right
in two runs of three, where Gemma managed at most three of the five.

Run it with:

```sh
LLM_MODEL=qwen/qwen3.6-35b-a3b LLM_REASONING_EFFORT=none \
  uv run python experiments/03_pydantic_ai/evaluate.py
```

(The Gemma numbers above were measured before the validator order fix.
Rerun them with Gemma loaded to compare like for like.)

### Claude Haiku 5.5

With `LLM_PROVIDER=anthropic LLM_MODEL=claude-haiku-5-5`, three runs passed
**5 of 5 each (100%)**, including the "SEV A" invoice question and the
60-user VPN case that the local models got wrong. Each run took about 16
seconds, and the three together cost about half a cent.

| Model | Average over three runs |
|---|---|
| `google/gemma-4-e4b` (local) | 53% |
| `qwen/qwen3.6-35b-a3b` (local, thinking off) | 73% |
| `claude-haiku-5-5` | 100% |

14 of the 15 cases took a single request. The one retry is worth a look:
for the vague database case, Haiku listed six questions in
`missing_info`, where the schema allows five. Claude's structured output
enforces the shape (fields, types, allowed values) but not limits such as
`maxItems`, `maxLength` or `minimum`: Pydantic AI moves those into the
field's description, where they're only a request, and Pydantic checks
them after the reply arrives. That's the second layer from the notebook
doing its job: it caught the extra question, and the retry fixed it.

Five cases are too few to call Haiku perfect: the next step is harder
cases (see Try next).

### A fairer comparison

Two things made the local models' numbers unfair: Qwen ran with its
thinking off (to avoid the empty replies above) while Haiku thinks, and
nobody had shown either local model what a good triage looks like. Two
flags on `evaluate.py` test both:

- `--output tool` sends the schema as a tool the model must call
  (`make_agent("tool")`), instead of as the response format. The answer
  arrives as a tool call, which LM Studio can't mistake for reasoning, so
  Qwen's thinking can stay on.
- `--examples` adds `EXAMPLES` from `triage.py` to the instructions: four
  worked cases, one per principle the evaluation checks (a customer's label
  doesn't set severity; one user with a connectivity error is networking at
  C; a group in one place is B; a vague case asks questions). None is one
  of the evaluation's cases, and a test makes sure of that, though they
  teach the principles the cases test, so the VPN case is no longer held out
  here.

Three runs each (six for Gemma with examples):

| Model | Setup | Average | Cases that gave up |
|---|---|---|---|
| `google/gemma-4-e4b` | as before | 60% | 3 |
| `google/gemma-4-e4b` | with examples | **80%** (4/5 every run) | 3 |
| `qwen/qwen3.6-35b-a3b` | thinking off | 73% | 0 |
| `qwen/qwen3.6-35b-a3b` | thinking on, tool output | 80% | 2 |
| `qwen/qwen3.6-35b-a3b` | thinking on, tool output, after the fix below | **100%** | 0 |
| `claude-haiku-5-5` | | 100% | 0 |

What it showed:

- **Worked examples made Gemma steady.** It got four cases right in every
  run, including the share and the "SEV A" invoice question it used to
  miss. The vague database case still fails: Gemma copied the example's
  `confidence: 0.4` but not its questions, and then repeated that answer
  after every "list what you need to know". It learned the surface of the
  example, not its point.
- **With thinking on, Qwen matches Haiku**, but slowly: about 2.5 minutes
  a run on this laptop, against 16 seconds for Haiku.
- **Tool output gives up the schema layer.** LM Studio enforces a response
  format while the model writes, but not a tool call's arguments. Qwen
  twice sent `affected_users` as the text `"null"` instead of `null`, and
  Pydantic's "unable to parse string as an integer" didn't tell it what to
  do instead, so it sent `"null"` again until its retries ran out. The fix
  is a `before` validator, `null_as_text_means_none()`, that repairs it:
  the meaning is clear, so there's nothing to send back.
- **The model reads your docstrings.** A model's docstring becomes its
  schema's description, which the model reads, and the output tool's
  description. `TriageSchema` now gives the model a description written for
  it (`json_schema_extra`), instead of notes written for developers.

## Try next

- **Examples for Qwen too.** Run `--output tool --examples` with Qwen, or
  `--examples` with Haiku, to see whether examples help models that already
  score well.
- **Harder cases.** Haiku and Qwen with thinking pass all five; add cases where the right answer
  is less obvious (an outage described without a number, a question that
  quotes an error code, a customer who exaggerates) and compare models
  again.
- **Compare models.** Run the evaluation with another `LLM_MODEL` and compare
  the experiments on the `triage-cases` dataset, case by case.
- **Add a case.** A case that's hard for the current rules: a group outage
  described without a number, or a question that mentions an error code.
- **Change the output mode.** Swap `NativeOutput` for `ToolOutput` (the
  schema becomes a tool the model must call) or `PromptedOutput` (the schema
  goes in the prompt), and compare how many retries each needs.
- **Drop the severity line** from `SYSTEM_PROMPT` and watch the invoice case,
  where the customer's own label says "SEV A".
