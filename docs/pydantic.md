# Pydantic and Pydantic AI for LLM output

A guide to read and practise before opening
[experiment 03](../experiments/03_pydantic_ai/). Part 1 covers Pydantic, the
library that checks data. Part 2 uses it on model output, first by hand, then
with Pydantic AI, the agent framework the experiment uses. Each section has
examples with their real output, and exercises with answers you can reveal.

Every example was run against the versions this project installs: Pydantic
2.14, the OpenAI SDK 3.37 and Pydantic AI 2.55. It's based on the Pydantic
notebook from the course, with its examples rewritten in the style of this
project and corrected where newer versions behave differently.

## Contents

0. [How to run the examples](#0-how-to-run-the-examples)
1. [The mental model: three layers](#1-the-mental-model-three-layers)

**Part 1. Pydantic**

2. [Models and required fields](#2-models-and-required-fields)
3. [Field constraints and reusable types](#3-field-constraints-and-reusable-types)
4. [Getting data in and out](#4-getting-data-in-and-out)
5. [Lax and strict conversion](#5-lax-and-strict-conversion)
6. [Extra fields](#6-extra-fields)
7. [Reading errors](#7-reading-errors)
8. [Validators](#8-validators)
9. [Nesting, inheritance and unions](#9-nesting-inheritance-and-unions)
10. [JSON Schema: what the model reads](#10-json-schema-what-the-model-reads)
11. [TypeAdapter: validating things that aren't models](#11-typeadapter-validating-things-that-arent-models)

**Part 2. Model output**

12. [Structured output by hand](#12-structured-output-by-hand)
13. [Pydantic AI: the same loop, done for you](#13-pydantic-ai-the-same-loop-done-for-you)
14. [Testing an agent without a model](#14-testing-an-agent-without-a-model)
15. [What the experiment learned](#15-what-the-experiment-learned)

**Reference**

16. [Cheat sheet](#16-cheat-sheet)
17. [Corrections to the course reference](#17-corrections-to-the-course-reference)
18. [Where it shows up in experiment 03](#18-where-it-shows-up-in-experiment-03)
19. [Sources](#19-sources)

---

## 0. How to run the examples

Parts 1 and most of Part 2 run without a model. `make install` installs
everything they need. Then either start a Python prompt:

```sh
uv run python
```

and paste the examples in one after another (each section builds on the
ones before only where it says so), or save them to a file and run it with
`uv run python file.py`.

Examples that call a model (sections 12 and the live part of 13) need LM
Studio running, as for the other experiments, and read the model and server
from `.env` through `Settings`.

The examples use the newer way of writing types: `str | None` instead of
`Optional[str]`, and `list[str]` instead of `List[str]`. They mean the same
thing; the course notebook uses the older spelling.

## 1. The mental model: three layers

Pydantic guards the point where untrusted data enters typed Python. You give
it a dict, a JSON string or a model's reply, and get back either an object
you can trust, or a `ValidationError` saying exactly what was wrong.
"Validation" includes conversion, so `"42"` can become `42` on the way in.

With model output there are three layers, and it helps to keep them apart:

| Layer | Where it runs | What it can enforce | Like... |
|---|---|---|---|
| **Schema** sent with the request | inside the model server, while the model writes (OpenAI's strict mode; LM Studio's grammars) | shape: keys, types, allowed values, some lengths and ranges | a form that won't submit without the required fields |
| **Pydantic validation** | your program, after the reply arrives | everything the schema can, plus rules across fields, checks against the source text, anything you can write in Python | a reviewer checking the form makes sense |
| **Retry with feedback** | your loop, or a framework's | sends the model the exact errors and asks for a fix | handing the form back with notes |

The schema is a mould: it makes malformed output impossible, but a perfectly
shaped answer can still be wrong. "Severity A" for one user's how-to question
fits the mould. The second layer checks meaning, and most of the interesting
work lives there. The third is rework.

## Part 1. Pydantic

## 2. Models and required fields

A model is a class that inherits from `BaseModel`, with a type on each field.
Whether a field is **required** depends only on whether it has a default.
Whether it can be **None** depends only on its type. These are separate
questions, and mixing them up is the most common surprise.

```python
from pydantic import BaseModel, ValidationError


class Ticket(BaseModel):
    title: str                 # required, can't be None
    owner: str | None          # required, but may be None
    tag: str | None = None     # may be left out, may be None
    retries: int = 0           # may be left out, can't be None


print(Ticket(title="VPN drops", owner=None))
# title='VPN drops' owner=None tag=None retries=0

try:
    Ticket(title="VPN drops")
except ValidationError as e:
    print(e)
# 1 validation error for Ticket
# owner
#   Field required [type=missing, input_value={'title': 'VPN drops'}, input_type=dict]
```

`owner` has no default, so leaving it out fails, even though `None` is
allowed. Remember this for section 10: OpenAI's strict mode marks every field
required in the schema it sends, so to the model every field is "always
present, maybe null".

## 3. Field constraints and reusable types

`Field()` adds limits (lengths, ranges) and a `description`. The description
matters more than it looks: it goes into the JSON Schema the model reads, so
it's part of the prompt. For limits you use in many places, `Annotated`
defines the type once.

```python
from typing import Annotated, Literal

from pydantic import BaseModel, Field, ValidationError

ShortText = Annotated[str, Field(min_length=10, max_length=200)]
Score = Annotated[float, Field(ge=0.0, le=1.0)]   # ge: >=, le: <=, gt: >, lt: <


class Finding(BaseModel):
    summary: ShortText
    confidence: Score
    severity: Literal["low", "medium", "high"]    # only these values
    tags: list[str] = Field(default_factory=list, max_length=3, description="At most three keywords.")


print(Finding(summary="Disk latency spikes every hour", confidence=0.8, severity="high"))
# summary='Disk latency spikes every hour' confidence=0.8 severity='high' tags=[]

try:
    Finding(summary="slow", confidence=1.4, severity="urgent", tags=["a", "b", "c", "d"])
except ValidationError as e:
    print(e.error_count(), "errors")
    for err in e.errors():
        print(" ", err["loc"], err["type"], "|", err["msg"])
# 4 errors
#   ('summary',) string_too_short | String should have at least 10 characters
#   ('confidence',) less_than_equal | Input should be less than or equal to 1
#   ('severity',) literal_error | Input should be 'low', 'medium' or 'high'
#   ('tags',) too_long | List should have at most 3 items after validation, not 4
```

Pydantic reports **every** problem at once, not just the first. That's what
makes its errors useful as feedback for a model.

Use `default_factory=list` for a list default. Pydantic copies a plain `[]`
default safely, so `tags: list[str] = []` also works, but `default_factory`
says what you mean and is safe in plain Python classes too, where a shared
`[]` default is a classic bug.

**Practice 3.** Write a `Receipt` model with `store` (2 to 50 characters),
`total` (more than 0) and `currency` (only `"USD"`, `"EUR"` or `"GBP"`, with
`"USD"` as the default). Check that `Receipt(store="X", total=0)` reports two
errors.

<details><summary>Answer</summary>

```python
class Receipt(BaseModel):
    store: str = Field(min_length=2, max_length=50)
    total: float = Field(gt=0)
    currency: Literal["USD", "EUR", "GBP"] = "USD"


try:
    Receipt(store="X", total=0)
except ValidationError as e:
    print(e.error_count())   # 2
```

</details>

## 4. Getting data in and out

| Direction | Method | Takes | Notes |
|---|---|---|---|
| In | `Model(**data)` | keyword arguments | can't take validation context (section 8) |
| In | `Model.model_validate(obj)` | a dict or object | takes `context=` and `strict=` |
| In | `Model.model_validate_json(text)` | a JSON `str` or `bytes` | parses and validates in one step; faster than `json.loads` first |
| In | `Model.model_validate_strings(d)` | a dict of strings | validates as if every value came from JSON |
| Out | `m.model_dump()` | | Python objects; `mode="json"` gives JSON-safe values |
| Out | `m.model_dump_json()` | | a JSON string |
| Trap | `Model.model_construct(**data)` | | **skips validation entirely**; only for data already checked |

```python
from datetime import date

from pydantic import BaseModel, Field


class Order(BaseModel):
    order_id: int = Field(ge=10000, le=99999)
    placed: date


o = Order.model_validate_json('{"order_id": "12345", "placed": "2026-10-01"}')
print(repr(o))                       # Order(order_id=12345, placed=datetime.date(2026, 10, 1))
print(o.model_dump())                # {'order_id': 12345, 'placed': datetime.date(2026, 10, 1)}
print(o.model_dump(mode="json"))     # {'order_id': 12345, 'placed': '2026-10-01'}
print(o.model_dump_json())           # {"order_id":12345,"placed":"2026-10-01"}

ghost = Order.model_construct(order_id=-5, placed="not a date")
print(repr(ghost))                   # Order(order_id=-5, placed='not a date')  <- nothing checked
```

For model output, use `model_validate_json` on the raw text, or
`model_validate` on a dict, so you can pass context.

## 5. Lax and strict conversion

By default Pydantic is **lax**: it converts values when the conversion is
unambiguous. **Strict** mode refuses anything that isn't already the right
type, except in JSON, where a string is the only way to write a date.

```python
from datetime import date

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class Lax(BaseModel):
    n: int
    flag: bool
    d: date


class Strict(BaseModel):
    model_config = ConfigDict(strict=True)
    n: int
    flag: bool
    d: date


print(repr(Lax(n="3", flag="yes", d="2026-10-23")))
# Lax(n=3, flag=True, d=datetime.date(2026, 10, 23))      "yes" counts as True

for label, make in [
    ("3.5 into int", lambda: Lax(n=3.5, flag=True, d="2026-10-23")),
    ("strict, from Python", lambda: Strict(n="3", flag="yes", d="2026-10-23")),
]:
    try:
        make()
    except ValidationError as e:
        print(label, "->", [err["type"] for err in e.errors()])
# 3.5 into int -> ['int_from_float']                       lax, but not lossy
# strict, from Python -> ['int_type', 'bool_type', 'date_type']

print(repr(Strict.model_validate_json('{"n": 3, "flag": true, "d": "2026-10-23"}')))
# Strict(n=3, flag=True, d=datetime.date(2026, 10, 23))   a date string is fine in JSON
```

Numbers are **not** turned into strings unless you ask:

```python
class Name(BaseModel):
    name: str


class NameCoerced(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)
    name: str


# Name(name=99999)          -> ValidationError: string_type
print(repr(NameCoerced(name=99999)))   # NameCoerced(name='99999')
```

For model output, lax is usually right: models sometimes quote numbers, and
`"0.8"` becoming `0.8` is harmless. Make single fields strict where a silent
conversion would hide a real problem:

```python
class PerField(BaseModel):
    count: int = Field(strict=True)   # "3" is an error here
    ratio: float                      # "0.5" is still fine here
```

## 6. Extra fields

What happens to keys the model doesn't define:

```python
from pydantic import BaseModel, ConfigDict, ValidationError


class Ignore(BaseModel):                      # the default
    a: int


class Forbid(BaseModel):
    model_config = ConfigDict(extra="forbid")
    a: int


class Allow(BaseModel):
    model_config = ConfigDict(extra="allow")
    a: int


payload = {"a": 1, "reasoning": "because"}
print(repr(Ignore(**payload)))                # Ignore(a=1)               dropped silently
print(Allow(**payload).model_extra)           # {'reasoning': 'because'}  kept separately
try:
    Forbid(**payload)
except ValidationError as e:
    print(e.errors()[0]["type"], e.errors()[0]["loc"])   # extra_forbidden ('reasoning',)
```

The default hides drift: if a model starts adding a `reasoning` key you
never asked for, you won't know. `extra="forbid"` is the check you control,
for every provider. (OpenAI's strict mode sends `additionalProperties: false`,
so extra keys can't come from there.)

## 7. Reading errors

Each entry in `e.errors()` has `type` (stable, good for code), `loc` (where:
a path), `msg` (for people) and `input` (what arrived). Two details matter:

1. `loc` is a **path**, so a nested error looks like `('items', 1, 'price')`.
   Printing only `loc[0]` loses where the problem is.
2. Errors from a model-level validator (section 8) have an **empty** `loc`,
   so `err["loc"][0]` raises `IndexError` on them.

A formatter that handles both:

```python
from pydantic import BaseModel, Field, ValidationError, model_validator


class Item(BaseModel):
    name: str
    price: float = Field(gt=0)


class Receipt(BaseModel):
    items: list[Item]
    total: float

    @model_validator(mode="after")
    def items_add_up(self):
        added = sum(item.price for item in self.items)
        if abs(added - self.total) > 0.01:
            raise ValueError(f"items add up to {added:.2f}, but the total is {self.total:.2f}")
        return self


def format_errors(e: ValidationError) -> str:
    lines = []
    for err in e.errors():
        where = ".".join(str(part) for part in err["loc"]) or "model"
        lines.append(f"  {where}: {err['msg']}")
    return "\n".join(lines)


try:
    Receipt(items=[{"name": "kibble", "price": 40}, {"name": "toy", "price": -3}], total=40)
except ValidationError as e:
    print(format_errors(e))
#   items.1.price: Input should be greater than 0

try:
    Receipt(items=[{"name": "kibble", "price": 40}, {"name": "toy", "price": 3}], total=40)
except ValidationError as e:
    print(e.errors()[0]["loc"])     # ()  <- empty
    print(format_errors(e))
#   model: Value error, items add up to 43.00, but the total is 40.00
```

Notice the first receipt never reached `items_add_up`. An "after" model
validator runs only once every field is valid, so rules across fields can
assume each field is already clean.

## 8. Validators

`Field` limits cover single values. Validators cover everything else.

| Kind | Mode | Receives | Use it for |
|---|---|---|---|
| Field | `before` | the raw input | cleaning messy values before the type check, like `"Sev A"` into `"A"` |
| Field | `after` (default) | the typed value | checks on a clean value; the safest choice |
| Field | `wrap` | the raw input and a `handler` | trying the normal path and recovering if it fails |
| Field | `plain` | the raw input | replacing Pydantic's checks for that field entirely |
| Model | `before` | the raw dict | reshaping the whole payload |
| Model | `after` | the built object, which it must return | rules across fields |
| Model | `wrap` | the raw input and a `handler` | logging or recovering around the whole model |

A validator fails by raising `ValueError`. Don't use `assert`: Python skips
asserts when run with `-O`.

You can attach a field validator with the `@field_validator` decorator, or
inline in the type with `Annotated[str, AfterValidator(fn)]`, which can be
reused and shows in the type itself.

### Validation context

Some checks need data that isn't in the payload, such as the customer's
original text, to check the model didn't invent anything. Pass it with
`context=` on any `model_validate*` call, and read it from `info.context`.
`Model(**data)` can't take context: one more reason to prefer
`model_validate`.

```python
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ValidationError, ValidationInfo, field_validator


def no_shouting(v: str) -> str:
    if v.isupper():
        raise ValueError("summary is all caps")
    return v


class Triage(BaseModel):
    severity: Literal["A", "B", "C"]
    summary: Annotated[str, AfterValidator(no_shouting)]
    quoted_error: str | None = None

    @field_validator("severity", mode="before")
    @classmethod
    def normalise_severity(cls, v):
        if isinstance(v, str):
            v = v.strip().upper().removeprefix("SEVERITY").removeprefix("SEV").strip()
        return v

    @field_validator("quoted_error")
    @classmethod
    def error_must_be_in_source(cls, v, info: ValidationInfo):
        source = (info.context or {}).get("source_text")
        if v is not None and source is not None and v not in source:
            raise ValueError(f"'{v}' does not appear in the source text")
        return v


source = "Users see 0x80070035 when mapping the share."
print(repr(Triage.model_validate(
    {"severity": " sev b", "summary": "Share mapping fails", "quoted_error": "0x80070035"},
    context={"source_text": source},
)))
# Triage(severity='B', summary='Share mapping fails', quoted_error='0x80070035')

try:
    Triage.model_validate(
        {"severity": "Sev A", "summary": "SHARE IS DOWN", "quoted_error": "0x80004005"},
        context={"source_text": source},
    )
except ValidationError as e:
    print(format_errors(e))     # from section 7
#   summary: Value error, summary is all caps
#   quoted_error: Value error, '0x80004005' does not appear in the source text
```

Note `"Sev A"` passed: the `before` validator cleaned it into `"A"` before
the `Literal` check ran.

### Repair, don't only reject

A validator can **fix** a value as well as reject it. When the right value
can be worked out by code, filling it in is better than sending the model
back to try again (section 15 explains why this mattered in the experiment):

```python
from pydantic import BaseModel, ValidationInfo, model_validator


class Case(BaseModel):
    affected_users: int | None = None

    @model_validator(mode="after")
    def fill_from_text(self, info: ValidationInfo):
        text = (info.context or {}).get("source_text", "")
        if self.affected_users is None and "4,000 users" in text:
            self.affected_users = 4000      # the code can read this fact itself
        return self


print(repr(Case.model_validate({}, context={"source_text": "all 4,000 users are locked out"})))
# Case(affected_users=4000)
```

(The experiment finds the number with a regular expression rather than a
fixed string; see `affected_users_matches_case()` in `triage.py`.)

**Practice 8.** Add to the `Receipt` from Practice 3 a field validator that
strips spaces from `store` and turns it into title case **before** the
length check, so `"  corner shop "` becomes `"Corner Shop"`. Then add a model
validator that rejects `currency="GBP"` when `total` is over 1000, with the
message "GBP receipts over 1000 need approval".

<details><summary>Answer</summary>

```python
class Receipt(BaseModel):
    store: str = Field(min_length=2, max_length=50)
    total: float = Field(gt=0)
    currency: Literal["USD", "EUR", "GBP"] = "USD"

    @field_validator("store", mode="before")
    @classmethod
    def tidy_store(cls, v):
        return v.strip().title() if isinstance(v, str) else v

    @model_validator(mode="after")
    def big_gbp_needs_approval(self):
        if self.currency == "GBP" and self.total > 1000:
            raise ValueError("GBP receipts over 1000 need approval")
        return self


print(Receipt(store="  corner shop ", total=5).store)   # Corner Shop
```

</details>

## 9. Nesting, inheritance and unions

**Inheritance:** a subclass adds fields (and validators) to its parent. The
experiment's `TriageValidated` inherits `TriageSchema` this way.

```python
class BaseTicket(BaseModel):
    title: str


class BugTicket(BaseTicket):
    steps_to_reproduce: list[str]


print(list(BugTicket.model_fields))     # ['title', 'steps_to_reproduce']
```

**Nesting:** a field can be another model (`items: list[Item]` in section 7),
and errors carry the full path.

**Discriminated unions** let the output be one of several shapes, with a
`Literal` tag deciding which. This is how agent "actions" are usually typed:
the model picks `search`, `reply` or `escalate`, each with its own fields. The
discriminator makes Pydantic check only the matching shape, so errors stay
short and specific.

```python
from typing import Annotated, Literal

from pydantic import BaseModel, Field, ValidationError


class Search(BaseModel):
    kind: Literal["search"]
    query: str


class Reply(BaseModel):
    kind: Literal["reply"]
    message: str = Field(min_length=1)


class Escalate(BaseModel):
    kind: Literal["escalate"]
    team: Literal["networking", "identity", "storage"]
    reason: str


Action = Annotated[Search | Reply | Escalate, Field(discriminator="kind")]


class Plan(BaseModel):
    steps: list[Action] = Field(min_length=1)


plan = Plan.model_validate({"steps": [
    {"kind": "search", "query": "AADSTS50076 conditional access"},
    {"kind": "escalate", "team": "identity", "reason": "tenant wide sign in failure"},
]})
print([type(step).__name__ for step in plan.steps])     # ['Search', 'Escalate']

for bad in [{"kind": "escalate", "team": "billing", "reason": "x"}, {"kind": "wave"}]:
    try:
        Plan.model_validate({"steps": [bad]})
    except ValidationError as e:
        print(format_errors(e))
#   steps.0.escalate.team: Input should be 'networking', 'identity' or 'storage'
#   steps.0: Input tag 'wave' found using 'kind' does not match any of the expected tags: 'search', 'reply', 'escalate'
```

**Practice 9.** Add a fourth action, `Close`, with `kind: Literal["close"]`
and `resolution: str` of at least 10 characters, and check that a plan with
`{"kind": "close", "resolution": "done"}` fails on `steps.0.close.resolution`.

<details><summary>Answer</summary>

```python
class Close(BaseModel):
    kind: Literal["close"]
    resolution: str = Field(min_length=10)


Action = Annotated[Search | Reply | Escalate | Close, Field(discriminator="kind")]


class Plan(BaseModel):
    steps: list[Action] = Field(min_length=1)


try:
    Plan.model_validate({"steps": [{"kind": "close", "resolution": "done"}]})
except ValidationError as e:
    print(format_errors(e))
#   steps.0.close.resolution: String should have at least 10 characters
```

`Plan` has to be defined again after `Action` changes, because it was built
with the old `Action`.

</details>

## 10. JSON Schema: what the model reads

`model_json_schema()` turns a model into JSON Schema, and that schema is the
real prompt for the structure: field names, `Literal` options and every
`description` reach the model. **Validators don't.** A rule that lives only
in a validator is invisible to the model until it fails, which is why the
experiment restates its rules in descriptions and in the system prompt.

```python
import json

print(json.dumps(Finding.model_json_schema(), indent=2))    # Finding from section 3
```

```json
{
  "properties": {
    "summary": {"maxLength": 200, "minLength": 10, "title": "Summary", "type": "string"},
    "confidence": {"maximum": 1.0, "minimum": 0.0, "title": "Confidence", "type": "number"},
    "severity": {"enum": ["low", "medium", "high"], "title": "Severity", "type": "string"},
    "tags": {
      "description": "At most three keywords.",
      "items": {"type": "string"}, "maxItems": 3, "title": "Tags", "type": "array"
    }
  },
  "required": ["summary", "confidence", "severity"],
  "title": "Finding",
  "type": "object"
}
```

OpenAI's SDK changes the schema before sending it in strict mode. You can
see this through a private helper (`openai.lib._pydantic`, so use it for
looking, not in real code):

```python
from openai.lib._pydantic import to_strict_json_schema

strict = to_strict_json_schema(Finding)
print(strict["required"], strict["additionalProperties"])
# ['summary', 'confidence', 'severity', 'tags'] False
```

Every field becomes required, `tags` included even though it has a default,
and extra keys are forbidden. The model always sends every field.

## 11. TypeAdapter: validating things that aren't models

Sometimes the model returns a bare list, or you want to validate against a
type like `Action` from section 9. `TypeAdapter` gives any type the same
methods a model has:

```python
from pydantic import TypeAdapter

ActionList = TypeAdapter(list[Action])
actions = ActionList.validate_json('[{"kind": "reply", "message": "Fixed after the policy rollback."}]')
print(actions)        # [Reply(kind='reply', message='Fixed after the policy rollback.')]
print(ActionList.json_schema()["type"])     # array
```

## Part 2. Model output

## 12. Structured output by hand

This is the loop a framework would otherwise hide, and it's worth writing
once. It needs LM Studio running.

The OpenAI SDK's `parse()` sends a Pydantic model as the response format
and gives back an instance of it:

```python
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from ai_playground.config import Settings

settings = Settings()
client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)


class Mood(BaseModel):
    mood: Literal["happy", "neutral", "upset"]
    reason: str = Field(min_length=5)


completion = client.chat.completions.parse(
    model=settings.llm_model,
    messages=[{"role": "user", "content": "Customer wrote: 'Third outage this week, I'm done.'"}],
    response_format=Mood,
    temperature=0,
)
parsed = completion.choices[0].message.parsed
print(repr(parsed))     # Mood(mood='upset', reason='service failure/unreliability')
```

Two things to know:

1. **`.parsed` is already a `Mood`, not a dict.** `Mood(**parsed)` raises
   `TypeError: Value after ** must be a mapping, not Mood`. To check it again
   with more rules, use `model_validate(parsed.model_dump(), context=...)`.
2. **`parse()` validates inside itself.** If the server lets something through
   that doesn't fit the class you sent, `parse()` raises `ValidationError`, so
   it goes inside its own `try`.

The retry loop, with Pydantic as the critic: send the errors back and ask
for a fix. This is the pattern Pydantic AI automates (section 13).

```python
def ask(messages: list, schema: type[BaseModel], validated: type[BaseModel], context: dict, attempts: int = 3):
    """Ask until the reply passes `validated`, sending the errors back each time."""
    for _ in range(attempts):
        try:
            completion = client.chat.completions.parse(
                model=settings.llm_model, messages=messages, response_format=schema, temperature=0
            )
        except ValidationError as e:        # didn't even fit the shape
            messages.append({"role": "user", "content": f"Your output did not match the schema:\n{format_errors(e)}\nTry again."})
            continue
        message = completion.choices[0].message
        try:
            return validated.model_validate(message.parsed.model_dump(), context=context)
        except ValidationError as e:        # the right shape, but it broke a rule
            messages.append({"role": "assistant", "content": message.content})
            messages.append({"role": "user", "content": f"Your answer failed these checks:\n{format_errors(e)}\nReturn a corrected answer."})
    return None
```

The schema class (`schema`) goes to the server; the stricter class with the
validators (`validated`) runs in your program. The experiment splits them the
same way: `TriageSchema` and `TriageValidated`.

## 13. Pydantic AI: the same loop, done for you

[Pydantic AI](https://pydantic.dev/docs/ai/core-concepts/agent/) is the agent framework from the
Pydantic team. An `Agent` bundles a model, instructions, an output type,
dependencies, tools and retry settings, and runs the loop from section 12.

| By hand (section 12) | Pydantic AI |
|---|---|
| `response_format=Schema` | `output_type=NativeOutput(Validated)` |
| `model_validate(..., context={...})` | `validation_context=lambda ctx: {...}` on the `Agent` |
| append the errors and call again | automatic: a `ValidationError` goes back as a retry prompt |
| an `attempts` counter | `retries={"output": 2}` (the default is **1**) |
| passing extra data around | `deps`, typed by `deps_type`, read through `RunContext` |
| rules that need outside data | `@agent.output_validator`, raising `ModelRetry` |

### The simplest agent

The examples in this section use `FunctionModel`, which stands in for a
model with a Python function (section 14 covers it). Swap it for a real
model to run them against LM Studio.

```python
from pydantic_ai import Agent, ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel


def reply(text):
    """A pretend model that always answers `text`."""
    return FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart(text)]))


agent = Agent(reply("Hello there!"))
result = agent.run_sync("Say hello")
print(repr(result.output), result.usage.requests)     # 'Hello there!' 1
```

Without an `output_type`, the output is text. `result.usage` (a property, not
a method, in this version) counts requests and tokens.

### A real model

The experiment builds its model from `Settings`, through LM Studio's
OpenAI-compatible server:

```python
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIChatModel(
    settings.llm_model,
    provider=OpenAIProvider(base_url=settings.llm_base_url, api_key=settings.llm_api_key),
)
agent = Agent(model)
```

For Claude, the same agent takes an `AnthropicModel`, which goes through
Anthropic's own API. (Anthropic also has an OpenAI-compatible endpoint, but
it ignores `response_format`, so `NativeOutput`'s schema wouldn't be
enforced.)

```python
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider

# Needs ANTHROPIC_API_KEY in .env; without it, AnthropicProvider raises UserError.
claude = AnthropicModel(
    "claude-haiku-5-5",
    provider=AnthropicProvider(api_key=settings.anthropic_api_key),
    settings=AnthropicModelSettings(anthropic_effort="medium"),
)
```

In this project, `LLM_PROVIDER=anthropic` makes experiment 03's
`make_model()` do this for you.

### Output modes

`output_type` can reach the model three ways, and the choice matters more
with small local models:

| Mode | How it works | Notes |
|---|---|---|
| `ToolOutput` (the default) | the schema becomes a tool the model must call | needs reliable tool calling |
| `NativeOutput` | the server's JSON response format | what LM Studio enforces with grammars; the experiment uses it |
| `PromptedOutput` | the schema is pasted into the instructions, and the text parsed | works with any model, least reliable |

### Retries, deps and output validators

A validation error, or a `ModelRetry` from an output validator, goes back to
the model as a retry prompt. `deps` carries data from your program (here, the
customer's support plan, which isn't in the message), and validators read it
through `RunContext`:

```python
import json
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel
from pydantic_ai import Agent, ModelResponse, ModelRetry, NativeOutput, RunContext, TextPart, capture_run_messages
from pydantic_ai.models.function import FunctionModel


@dataclass
class Deps:
    plan: Literal["unified", "standard"]


class Answer(BaseModel):
    severity: Literal["A", "B", "C"]
    escalate: bool


# A pretend model that forgets to escalate the first time.
answers = iter([{"severity": "B", "escalate": False}, {"severity": "B", "escalate": True}])
pretend = FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart(json.dumps(next(answers)))]))

triage = Agent(
    pretend,
    deps_type=Deps,
    output_type=NativeOutput(Answer),
    instructions="Triage the case.",
    retries={"output": 2},
)


@triage.output_validator
def unified_escalates(ctx: RunContext[Deps], output: Answer) -> Answer:
    if ctx.deps.plan == "unified" and output.severity in ("A", "B") and not output.escalate:
        raise ModelRetry("Unified plan: severity A or B must escalate.")
    return output


with capture_run_messages() as messages:
    result = triage.run_sync("Half the office is offline.", deps=Deps("unified"))
print(repr(result.output))       # Answer(severity='B', escalate=True)

for message in messages:
    for part in message.parts:
        print(message.kind, part.part_kind, repr(getattr(part, "content", ""))[:60])
# request user-prompt 'Half the office is offline.'
# response text '{"severity": "B", "escalate": false}'
# request retry-prompt 'Unified plan: severity A or B must escalate.'
# response text '{"severity": "B", "escalate": true}'
```

`capture_run_messages()` (or `result.all_messages()`) shows the exact
exchange, and it looks like the loop from section 12. What the model reads
on a retry:

```
Validation feedback:
Unified plan: severity A or B must escalate.
Fix the errors and try again.
```

and for a `ValidationError` from the output type, the error list as JSON:

~~~
1 validation error:
```json
[{"type": "greater_than_equal", "loc": ["count"], "msg": "Input should be greater than or equal to 1"}]
```
Fix the errors and try again.
~~~

When the retries run out, the run raises `UnexpectedModelBehavior: Exceeded
maximum output retries (2)`.

### When retries can't help

A retry only helps if the prompt tells the model something it can act on.
Two ways it can't, both seen in experiment 03:

- **The answer never reaches Pydantic AI.** A thinking model such as Qwen 3
  can have its whole JSON answer filed as "thinking" by LM Studio, with an
  empty reply. Pydantic AI then sends a generic "Please return text.", the
  model repeats itself, and every retry is wasted. If the retry prompts in
  `capture_run_messages()` say "Please return text.", look at the response
  parts: a `thinking` part with the JSON in it is this problem. The fix is
  turning thinking off (`openai_reasoning_effort="none"`, or
  `LLM_REASONING_EFFORT=none` in this project).
- **The feedback names the wrong field.** Model validators run in the order
  they're written, and the first to fail is the one the model hears about.
  Put the checks that can say precisely what's wrong first.

### Validation context

`validation_context` passes data to the output type's validators (section 8),
built from the run's deps:

```python
from pydantic import ValidationInfo, field_validator


class Codes(BaseModel):
    codes: list[str]

    @field_validator("codes")
    @classmethod
    def in_text(cls, codes, info: ValidationInfo):
        text = (info.context or {}).get("text", "")
        missing = [code for code in codes if code not in text]
        if missing:
            raise ValueError(f"not in the case: {missing}")
        return codes


agent = Agent(
    model,                                   # a pretend or real model
    deps_type=str,
    output_type=NativeOutput(Codes),
    validation_context=lambda ctx: {"text": ctx.deps},
)
# A model that first answers ["E42", "E99"] gets "not in the case: ['E99']"
# back, and then answers ["E42"].
```

### `run_sync` or `await run`

In a script, `agent.run_sync(...)` is fine. In a Jupyter notebook, which
already runs an event loop, `run_sync` fails with "This event loop is already
running"; use `await agent.run(...)` there instead. That's why the course
notebook uses `await`, and the experiment, a script, doesn't.

### Tracing

`setup_tracing(pydantic_ai=True)` records Pydantic AI's own spans: one
`invoke_agent` span per run, with a `chat` span for each request, so retries
are easy to see. See
[framework spans](tracing.md#framework-spans-pydantic-ai).

**Practice 13.** Give `triage` above a second output validator that rejects
severity A for a `"standard"` plan with the message "Standard plans can't be
severity A", and run it with a pretend model that answers A then B. How many
requests does it take?

<details><summary>Answer</summary>

```python
@triage.output_validator
def standard_never_a(ctx: RunContext[Deps], output: Answer) -> Answer:
    if ctx.deps.plan == "standard" and output.severity == "A":
        raise ModelRetry("Standard plans can't be severity A")
    return output


answers = iter([{"severity": "A", "escalate": True}, {"severity": "B", "escalate": False}])
with triage.override(model=FunctionModel(lambda m, i: ModelResponse(parts=[TextPart(json.dumps(next(answers)))]))):
    result = triage.run_sync("Something broke.", deps=Deps("standard"))
print(result.output, result.usage.requests)     # severity='B' escalate=False 2
```

</details>

## 14. Testing an agent without a model

Three pieces make agent tests fast and offline, as in the experiment's
[`test_triage.py`](../experiments/03_pydantic_ai/test_triage.py):

- **`FunctionModel`** replaces the model with a Python function that sees the
  message history and returns a reply. Counting the requests in the history
  lets it answer differently on a retry.
- **`agent.override(model=...)`** swaps it in without changing the agent.
- **`models.ALLOW_MODEL_REQUESTS = False`** makes any accidental real call
  fail: `RuntimeError: Model requests are not allowed, since
  ALLOW_MODEL_REQUESTS is False`.

```python
from pydantic_ai import models

models.ALLOW_MODEL_REQUESTS = False


def answers(*replies):
    """A pretend model that gives these replies in turn, one per request."""
    def model(messages, info):
        requests = sum(1 for message in messages if message.kind == "request")
        return ModelResponse(parts=[TextPart(json.dumps(replies[requests - 1]))])
    return FunctionModel(model)


with triage.override(model=answers({"severity": "B", "escalate": False}, {"severity": "B", "escalate": True})):
    result = triage.run_sync("Half the office is offline.", deps=Deps("unified"))
assert result.output.escalate
```

The validators themselves need no agent at all: call `model_validate` with
hand-written dicts, one rule broken per test. Prove the checks work before
spending a single token.

## 15. What the experiment learned

[Experiment 03](../experiments/03_pydantic_ai/README.md#what-it-found)
turned the triage into an evaluation. Four lessons from it:

1. **Valid isn't right.** Every case ended with a valid triage, and several
   were wrong. Validators check consistency; only an evaluation with expected
   answers checks correctness.
2. **Make rules match the definitions you already wrote.** The schema said
   "C: single user, how-to question", but the validator only rejected A for
   them, so B slipped through. Every rule in a `description` should have a
   validator behind it.
3. **A small model often ignores feedback.** Replaying a failing case showed
   it sending exactly the same answer after each retry. A rule it can't
   satisfy turns a wrong-but-valid answer into a run that gives up.
4. **Repair what code can work out, and name the wrong field.** Filling in
   `affected_users` from the text (section 8) made it possible to say
   precisely "the impact can't be single_user when 4,000 users are affected".
   Pointing at the field that's actually wrong gives the model something it
   can fix.
5. **The order of validators decides the feedback.** Pydantic stops at the
   first model validator that fails. When a vaguer check ran first, the
   precise one never did, and Qwen spent its retries refusing a wrong
   suggestion. Running the precise check first fixed it on the first retry.
6. **Check what the model actually returned before blaming the rules.** A
   thinking model's answer can arrive as reasoning with an empty reply, and
   no number of retries will fix that (section 13).

## Reference

## 16. Cheat sheet

| Need | Reach for |
|---|---|
| Required, but may be None | `x: int \| None` with no default |
| Reusable limits | `Annotated[str, Field(min_length=10)]` |
| Clean messy input first | `@field_validator(..., mode="before")` |
| A rule across fields | `@model_validator(mode="after")`, returning `self` |
| A check against outside data | `model_validate(data, context={...})` and `info.context` |
| Fill in a value code can work out | an `after` model validator that sets the field |
| Catch unexpected keys | `ConfigDict(extra="forbid")` |
| No silent conversion | `Field(strict=True)` or `ConfigDict(strict=True)` |
| One of several shapes | `Annotated[A \| B, Field(discriminator="kind")]` |
| Validate a bare list | `TypeAdapter(list[Model])` |
| See what the model reads | `Model.model_json_schema()` |
| Structured output with retries | `Agent(output_type=NativeOutput(Model), retries={"output": 2})` |
| A rule that needs program data | `deps_type`, `@agent.output_validator` and `ModelRetry` |
| Test an agent offline | `FunctionModel`, `agent.override()`, `models.ALLOW_MODEL_REQUESTS = False` |
| See what was sent and received | `capture_run_messages()` or `result.all_messages()` |

## 17. Corrections to the course reference

The course's `docs.md` reference is out of date in places. Checked against
Pydantic 2.14 and the OpenAI SDK 3.37:

| `docs.md` says | What actually happens |
|---|---|
| Pattern 6: `TestModel(name=99999)` gives `"99999"` | It fails with `string_type`. It needs `ConfigDict(coerce_numbers_to_str=True)` (section 5) |
| `.parsed` "is a Python dictionary" | It's an instance of your model, and `MyModel(**parsed)` raises `TypeError` (section 12) |
| Print errors with `error['loc'][0]` | `IndexError` for model validator errors, and it drops the path of nested ones (section 7) |
| `action_items: List[str] = []` | Works, but `Field(default_factory=list)` says what you mean (section 3) |
| `client.beta.chat.completions.parse` | Still works; `client.chat.completions.parse` is the path in current SDKs |
| `parse()` output is "already validated" | Only against the class you sent. Rules across fields and against the source need your own validators |

## 18. Where it shows up in experiment 03

| Concept | Section | In the experiment |
|---|---|---|
| Two classes: shape and meaning | 1, 9, 12 | `TriageSchema` and `TriageValidated` in `triage.py` |
| `Literal`, limits, descriptions as prompt | 3, 10 | every field of `TriageSchema` |
| Validation context | 8 | `codes_must_appear_in_case()`, `affected_users_matches_case()` |
| Rules across fields | 8 | `severity_matches_impact()`, `low_confidence_needs_questions()` |
| Repairing a value | 8, 15 | `affected_users_matches_case()` fills in a stated count |
| `NativeOutput`, retries, `validation_context` | 13 | `triage_agent` |
| `deps` and an output validator | 13 | `CaseDeps` and `unified_plan_escalates()` |
| `capture_run_messages()` | 13 | `main.py` counts each case's requests |
| Offline tests | 14 | `test_triage.py` |
| Valid isn't right | 15 | `evaluate.py` and `triage_grader.py` |

## 19. Sources

- [Pydantic: models](https://pydantic.dev/docs/validation/latest/concepts/models/)
- [Pydantic: validators](https://pydantic.dev/docs/validation/latest/concepts/validators/)
- [Pydantic: conversion table, lax and strict](https://pydantic.dev/docs/validation/latest/concepts/conversion_table/)
- [Pydantic AI: agents](https://pydantic.dev/docs/ai/core-concepts/agent/)
- [Pydantic AI: output](https://pydantic.dev/docs/ai/core-concepts/output/)
- [Pydantic AI: testing](https://pydantic.dev/docs/ai/guides/testing/)
- [LM Studio: structured output](https://lmstudio.ai/docs/developer/openai-compat/structured-output)
- [OpenAI: structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
