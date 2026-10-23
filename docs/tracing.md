# Tracing the agent

This guide explains how tracing works in this project: what traces and spans
are, how the code creates them, how they reach Phoenix and Langfuse, and how
to use them to understand what the agent did. It refers to the code in
[`tracing.py`](../src/ai_playground/tracing.py) and
[`llm.py`](../src/ai_playground/llm.py), as used by
[experiment 01](../experiments/01_tool_calling/main.py).

For setup steps (Docker, `.env`, starting the tools), see the
[README](../README.md#tracing-with-phoenix-and-langfuse).

## Contents

1. [Why trace an agent?](#1-why-trace-an-agent)
2. [Traces and spans](#2-traces-and-spans)
3. [The trace this project produces](#3-the-trace-this-project-produces)
4. [How a span becomes a child of another span](#4-how-a-span-becomes-a-child-of-another-span)
5. [Automatic and manual spans](#5-automatic-and-manual-spans)
6. [Attributes: what a span records](#6-attributes-what-a-span-records)
7. [How spans get from Python to Phoenix and Langfuse](#7-how-spans-get-from-python-to-phoenix-and-langfuse)
8. [Phoenix and Langfuse side by side](#8-phoenix-and-langfuse-side-by-side)
9. [What runs in Docker](#9-what-runs-in-docker)
10. [Exercises](#10-exercises)
11. [Troubleshooting](#11-troubleshooting)
12. [Glossary](#12-glossary)

---

## 1. Why trace an agent?

A normal program follows the path you wrote. An agent doesn't: the model
decides which tools to call, with what arguments, and how many times to go
round the loop. When the answer is wrong, the final text alone doesn't tell
you why. You need to know:

- What did the model receive, including the tool descriptions?
- Which tools did it ask for, with which arguments?
- What did each tool return?
- How many trips to the model did it take, and how long did each one take?
- How many tokens did each call use?

`print()` statements can answer some of this, but they get lost in the
terminal, they don't show which step happened inside which, and they
disappear when the program ends. A trace records all of it, as a tree you can
open and explore later.

## 2. Traces and spans

A **span** is one step of work. It has:

- a **name**, such as `turn 1` or `get_current_time`;
- a **start time** and an **end time**, so you can see how long it took;
- **attributes**: details we attach, such as the input and output;
- a **status**: OK, or an error with the exception that caused it;
- a **parent**: the span it happened inside, if any.

A **trace** is all the spans from one run, connected by their parents into a
tree. The span with no parent is the **root span**. In this project, each call
to `call_llm()` produces one trace, and its root span is `agent`.

Every span carries a **trace ID** (shared by all spans in the trace) and its
own **span ID**. A child span stores its parent's span ID. That is all the
tools need to rebuild the tree.

## 3. The trace this project produces

Running experiment 01 asks: *"What is the current local time and the current time
in UTC?"* The trace looks like this:

```
agent                                   AGENT   the whole run
├── turn 1                              CHAIN   first trip to the model
│   ├── Response                        LLM     model asks for two tools
│   ├── get_current_time                TOOL    returns "23:56:14 (11:56 PM)"
│   └── get_current_time_in_timezone    TOOL    {"timezone": "UTC"} -> "03:56:14 (3:56 AM)"
└── turn 2                              CHAIN   second trip to the model
    └── Response                        LLM     model writes the final answer
```

Reading it from the top:

1. **`agent`** starts when `call_llm()` begins and ends when it returns. Its
   input is the prompt and its output is the final answer.
2. **`turn 1`** is one pass through the `while True` loop in `call_llm()`.
3. Inside it, **`Response`** is the call to the model. The model doesn't
   answer yet: it replies with two function calls.
4. The code runs both tools. Each gets its own span with the arguments the
   model sent and the value the tool returned.
5. **`turn 2`** sends the history, now including the tool results, back to
   the model. This time the model replies with text and no tool calls, so the
   loop ends.

Each line in the tree comes from a specific place in the code:

| Span | Created by | Where |
|---|---|---|
| `agent` | `agent_span(prompt)` | `call_llm()` in `llm.py` |
| `turn N` | `turn_span(turn)` | the `while True` loop in `call_llm()` |
| `Response` | the OpenAI instrumentor, automatically | `client.responses.create(...)` |
| tool spans | `tool_span(call.name, call.arguments)` | `run_tool()` in `llm.py` |

The number of turns isn't fixed. If the model asked for one tool, then
another, then answered, you would see three turns. Counting turns is one of
the simplest ways to see whether an agent is working efficiently.

## 4. How a span becomes a child of another span

Nowhere in the code do we say "`turn 1` is the parent of
`get_current_time`". It happens because of `with`:

```python
with agent_span(prompt) as agent:              # agent span starts
    ...
    with turn_span(turn) as current_turn:      # starts inside agent
        response = client.responses.create(...)    # starts inside turn
        ...
        history.append(run_tool(call, tools_by_name))
            # inside run_tool:
            # with tool_span(...) as span:     # starts inside turn
```

OpenTelemetry keeps track of a **current span**. `start_as_current_span()`,
which our helpers call, does two things:

1. It creates a new span whose parent is whatever span is current right now.
2. It makes the new span the current one until its `with` block ends, and
   then puts the previous one back.

So whatever span is current when a new one starts becomes its parent. This
works across function calls: `run_tool()` doesn't receive the turn span as an
argument, but the turn span is current when `run_tool()` runs, so the tool
span becomes its child. The automatic `Response` span works the same way:
the OpenAI instrumentor starts its span inside our turn span, so it becomes a
child without us passing anything to it.

A span ends when its `with` block ends, including when the block is left
early by `return` or by an exception. That's why `return response` inside
both `with` blocks is safe: both spans still end and are recorded.

### What `@contextmanager` does

The helpers in `tracing.py` look like this:

```python
@contextmanager
def turn_span(number: int):
    with tracer.start_as_current_span(f"turn {number}") as span:
        span.set_attribute(SPAN_KIND, "CHAIN")
        yield span
```

`@contextmanager` lets a function be used with `with`. When the
`with turn_span(...)` block starts, the function runs up to `yield`, so the
span starts and gets its attributes. The value after `yield` is what you get
in `as current_turn`. When the block ends, the function continues after
`yield`, leaving the inner `with`, which ends the span.

## 5. Automatic and manual spans

There are two ways spans get created, and this project uses both.

**Automatic spans (auto-instrumentation).** In `setup_tracing()`:

```python
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
```

This wraps the OpenAI library's own methods so that every
`client.responses.create(...)` call is recorded as a span, with no change to
our code. The instrumentor knows the shape of OpenAI requests and responses,
so it records far more than we would by hand: the full input messages, the
tool definitions, the model's output, the model name and the token counts.
It comes from [OpenInference](https://github.com/Arize-ai/openinference),
which has instrumentors for many LLM libraries.

**Manual spans.** The OpenAI library knows nothing about our agent loop, our
turns or our tools: those exist only in our code. So we create those spans
ourselves with `agent_span`, `turn_span` and `tool_span`.

Most real projects mix the two: libraries are traced automatically, and the
steps specific to your application are traced by hand.

## 6. Attributes: what a span records

Attributes are key/value pairs on a span. Any name works, but tools only
display a span nicely if they recognise the names. We use the names from
**OpenInference**, which Phoenix created and Langfuse also understands. They
are defined once, at the top of the helpers section in `tracing.py`:

| Attribute | Example | Meaning |
|---|---|---|
| `openinference.span.kind` | `AGENT`, `CHAIN`, `TOOL`, `LLM` | What kind of step this is. Tools use it for the label and icon. |
| `input.value` | the prompt; `{"timezone":"UTC"}` | What went into the step. |
| `output.value` | the final answer; `"03:56:14 (3:56 AM)"` | What came out of the step. |
| `tool.name` | `get_current_time` | Which tool ran (tool spans only). |

The span kinds we use:

- **AGENT**: a whole agent run, from question to answer.
- **CHAIN**: a group of related steps; here, one turn.
- **TOOL**: one tool call.
- **LLM**: one model call. The instrumentor sets this on `Response` spans.

The automatic `Response` spans carry many more attributes, such as
`llm.model_name`, `llm.token_count.prompt`, `llm.token_count.completion` and
`llm.input_messages.*`. Click a `Response` span in Phoenix to see them all.

### Other kinds of step, as the playground grows

OpenInference defines more kinds than this experiment uses. All of them are
listed in `SpanKind` in `tracing.py`, and the general `span()` helper creates
a span of any kind:

```python
from ai_playground.tracing import SpanKind, set_output, span

with span("search docs", SpanKind.RETRIEVER, question) as current:
    documents = search(question)
    set_output(current, json.dumps(documents))
```

`agent_span`, `turn_span` and `tool_span` are just shortcuts built on
`span()`. What later experiments are likely to need:

| Building | Spans to expect | What to do |
|---|---|---|
| RAG (retrieval) | EMBEDDING, RETRIEVER, RERANKER | Use `span()` with that kind around each step. If the library has an OpenInference instrumentor, add it in `setup_tracing()` and the spans appear by themselves. |
| Guardrails | GUARDRAIL | `span()` around each check, with the verdict as the output. |
| Evaluations | EVALUATOR | Already done: `evals.py` puts each grade in an EVALUATOR span. |
| MCP tools | TOOL | A tool called through MCP still goes through `run_tool()`, so it gets a tool span. When the MCP server is a separate program, OpenInference's MCP instrumentor passes the trace along, so spans the server creates join the same trace instead of starting a new one. |
| Several agents | AGENT inside AGENT | Nothing to do: an agent run inside another agent's tool becomes its child automatically (section 4). |
| Chat with memory | many traces, one per message | Group them into a *session* by setting a `session.id` attribute, which both tools use to show a whole conversation together. |

In each case the new code goes in `tracing.py`, and the rest of the code
keeps calling helpers.

### Why the names live only in `tracing.py`

`llm.py` never writes an attribute name. It calls helpers like
`set_output(span, result)`. If we later switch to a different naming scheme,
for example OpenTelemetry's newer `gen_ai.*` names for AI applications, only
`tracing.py` changes. Production teams use small helpers like these for the
same reason: every span of a given kind gets the same attributes, in the same
format.

## 7. How spans get from Python to Phoenix and Langfuse

`setup_tracing()` builds a small pipeline. From the outside in:

```
                 TracerProvider  (resource: openinference.project.name = agentic-ai-course)
                 │
spans ──────────►├── BatchSpanProcessor ──► OTLPSpanExporter ──► Phoenix   localhost:6006/v1/traces
                 │
                 └── BatchSpanProcessor ──► OTLPSpanExporter ──► Langfuse  localhost:3000/api/public/otel/v1/traces
```

- **TracerProvider** is the centre of OpenTelemetry. `trace.get_tracer()`
  returns tracers that report to it, and every span that ends is handed to
  each of its processors. Because there are two processors, **every span goes
  to both tools**.
- **Resource** describes who is sending the spans. Phoenix reads the
  `openinference.project.name` resource attribute to choose the project.
  Langfuse ignores it and uses the project that the API keys belong to.
- **BatchSpanProcessor** collects finished spans and sends them in groups,
  in the background, so tracing doesn't slow the agent down.
- **OTLPSpanExporter** sends a batch over HTTP using **OTLP**, the
  OpenTelemetry Protocol. Every OpenTelemetry-compatible tool accepts OTLP,
  so the exporter code for Phoenix and Langfuse is the same apart from the
  address. Langfuse also needs two headers (see `langfuse_headers()`): the
  API keys in `Authorization`, and `x-langfuse-ingestion-version: 4`, which
  makes new spans readable through its API straight away.

`trace.set_tracer_provider(tracer_provider)` makes this provider the one the
whole program uses, which is how the spans created in `llm.py` find it.

### Things that follow from this design

- **Spans are sent when they end, not when they start.** Children end before
  their parents, so the `agent` span is the last to be sent.
- **Spans are sent in the background.** When the program exits, OpenTelemetry
  sends whatever is still waiting (it registers a shutdown handler for this).
  A program that crashes hard can lose its last spans.
- **Without `setup_tracing()`, nothing is recorded.** `trace.get_tracer()`
  then returns a tracer whose spans do nothing. This is why the tests run
  without Phoenix or Langfuse.
- **Adding a third tool is one line in `setup_tracing()`**: another
  `add_span_processor(...)` with that tool's exporter. Nothing else changes.

### Where settings come from

All settings live in one class, `Settings` in
[`config.py`](../src/ai_playground/config.py). Each field can be set by an
environment variable of the same name in capitals, or by a line in `.env`.
The tracing settings are:

| Variable | Default | Used for |
|---|---|---|
| `TRACE_PROJECT_NAME` | `agentic-ai-course` | The Phoenix project traces are grouped under |
| `PHOENIX_COLLECTOR_ENDPOINT` | `http://localhost:6006` | Phoenix address |
| `LANGFUSE_HOST` | `http://localhost:3000` | Langfuse address |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | none | Langfuse API keys. Without them, Langfuse is skipped. |

Langfuse ignores `TRACE_PROJECT_NAME`: its project is the one the API keys
belong to.

## 8. Phoenix and Langfuse side by side

Both receive exactly the same spans. What differs is how they name and show
them:

| Our span | Phoenix shows | Langfuse shows |
|---|---|---|
| `agent` | AGENT | AGENT |
| `turn 1`, `turn 2` | CHAIN | CHAIN |
| `Response` | LLM | GENERATION |
| tool spans | TOOL | TOOL |

Langfuse calls model calls **generations**. It's the same thing as Phoenix's
**LLM** spans: one request to the model and its reply, with the model name
and token usage.

Some things to compare as you click around:

- **The tree.** Both show the same parent/child structure from section 3.
- **Model calls.** Open a `Response` span in each. Both show the messages
  sent to the model and its reply; compare how each one lays them out.
- **Timing.** Both show how long each span took. Most of the time is spent
  in the `Response` spans: tools that read the clock take microseconds.
- **Tokens.** Both show token counts on model calls. Compare turn 1 and
  turn 2: the second sends more input, because the history now includes the
  first reply and the tool results.

### Scores from evaluations

[Experiment 02](../experiments/02_evaluation/) grades the agent's answers
and attaches each verdict to that case's trace, using
[`scores.py`](../src/ai_playground/scores.py). Scores aren't spans: both
tools store them separately and receive them through their own APIs, after
the spans have been sent.

| | Phoenix | Langfuse |
|---|---|---|
| Called | annotation | score |
| Sent to | `POST /v1/trace_annotations` | `POST /api/public/scores` |
| What's stored | label (`pass`/`fail`), score (1 or 0), explanation | value (1 or 0, as a boolean), comment |
| Where to see it | the trace's annotations; filter or add a column for `correct` | the trace's scores, and the Scores page |

Filtering by score is the quickest way to find failures: show only traces
where `correct` is `fail` (Phoenix) or `0` (Langfuse), then open one to see
what the model did.

Other features differ more. Langfuse has prompt management and cost
dashboards; both can store datasets of test cases. Those are worth exploring
once traces and scores feel familiar.

## 9. What runs in Docker

`docker compose up -d` starts seven containers:

| Container | What it does |
|---|---|
| `phoenix` | All of Phoenix: the UI, the OTLP endpoint and storage, in one container. Data is kept in the `phoenix_data` volume. |
| `langfuse-web` | Langfuse's UI and API, including the OTLP endpoint we send to. |
| `langfuse-worker` | Processes incoming traces in the background and writes them to storage. |
| `postgres` | Stores users, organisations, projects and API keys. |
| `clickhouse` | Stores traces and observations. It's a database built for fast queries over large amounts of event data. |
| `redis` | A queue and cache between `langfuse-web` and `langfuse-worker`. |
| `minio` | S3-compatible file storage for raw incoming events and media. |

Phoenix is one container because it's designed to be simple to run locally.
Langfuse is split into parts because it's designed to scale: in production,
the web and worker containers can be scaled separately, and each database can
be replaced by a managed service. Running it locally is a small taste of how
a production observability system is built.

A trace you send to Langfuse goes through several of these: `langfuse-web`
accepts it and stores the raw event in MinIO, a job is queued in Redis,
`langfuse-worker` picks it up and writes it to ClickHouse, and the UI reads it
from there. This is why a new trace can take a few seconds to appear in
Langfuse, while it shows up in Phoenix almost immediately.

The Langfuse setup is the official
[`docker-compose.yml`](../docker/langfuse/docker-compose.yml), unchanged,
included from the project's [`docker-compose.yml`](../docker-compose.yml).
Its secrets come from `.env`. On first start, the `LANGFUSE_INIT_*` settings
create the organisation, project, user and API keys, so there's no sign-up
step.

### Where the data lives, and moving it to another machine

Everything Phoenix and Langfuse store (traces, scores, the Langfuse user and
API keys) is in six Docker **volumes**, named `ai-playground_<name>`:
`phoenix_data` and five `langfuse_*` ones. A volume is storage Docker
manages, kept when a container is removed or recreated. On macOS, Docker
runs inside Colima's Linux virtual machine, so the volumes are on its disk
image (`~/.colima/_lima/_disks/colima/datadisk`), not in this folder.

| Command | Data |
|---|---|
| `docker compose down` | Kept |
| `docker compose down -v` | Deleted (`-v` removes the volumes) |
| `colima delete` | Deleted, along with everything else in Docker |

None of it is in git, so on another machine you start with empty tools. To
take it with you:

```sh
make backup                                     # on this machine
# copy the new backups/<date-time>/ folder to the other machine, privately
make restore BACKUP=backups/<date-time>        # on the other machine
```

[`backup.sh`](../docker/backup.sh) stops the containers, so no database is
half-way through writing, packs each volume into a `.tar.gz` file, and
starts them again. [`restore.sh`](../docker/restore.sh) replaces the volumes'
contents with the backup's, after asking.

Each backup also holds two other files:

- **`.env`**, because Langfuse encrypts some data with `ENCRYPTION_KEY` and
  `SALT`, and the databases expect the passwords they were created with.
  With a different `.env`, the restored data can't be read, so the restore
  stops and says so. The backup is a secret: never commit or share it.
  `backups/` is ignored by git.
- **`images.txt`**, the image versions. A database's files only work with
  the version that wrote them, so the restore warns if the versions differ.

You only need this for the history of past runs. The code, evaluation cases
and graders are in git, so any run can be repeated anywhere.

## 10. Exercises

Each one changes something small and shows how it appears in the trace. Run
`uv run python experiments/01_tool_calling/main.py` after each change, then
find the newest trace in both tools.

1. **Read one trace end to end.** Without changing anything, open the newest
   trace in Phoenix. Find the tool definitions the model received, the exact
   arguments it chose for `get_current_time_in_timezone`, and the history it
   was sent in turn 2. Then find the same things in Langfuse.

2. **Find who made a mistake.** Models get things wrong even when their
   tools are right. When the tools returned only a 24-hour time,
   `get_current_time` once returned `00:06:43` (just after midnight) and the
   model answered "12:06 PM". The trace showed it at once: the tool span's
   output was correct, so the error was in the model's final `Response`.
   That's why the tools now also state AM or PM (see
   [experiment 02](../experiments/02_evaluation/README.md#what-it-found)).
   Whenever an answer looks wrong, compare each tool span's output with what
   the model said. To see the old mistake again, make `format_time` in
   `tools.py` return only the 24-hour time, and ask for the time in a city
   where it's between midnight and noon.

3. **See an error.** In the experiment's `main.py`, change the question to ask for the time on
   `Mars/Olympus`. The tool raises an error for the unknown timezone. In the
   trace, the tool span is marked as an error with the exception attached,
   and the error also marks `turn 1` and `agent`, because it passed through
   them on its way out. Errors are where tracing pays off most: you see
   exactly which step failed and what it was given.

4. **Change the number of turns.** Ask a question that needs one tool's
   answer before calling the next, for example: *"What time is it in UTC?
   Then tell me the time in the timezone that is 9 hours ahead of UTC."*
   Count the turns. Then try a question that needs no tools at all, such as
   *"What is 2 + 2?"*, and see a trace with a single turn.

5. **Add a tool and watch it appear.** Write a new function in `tools.py`
   with a docstring and type hints, for example
   `def add(a: int, b: int) -> int`, and add it to `tools=[...]` in
   `main.py`. Its definition now shows up in the `Response` spans' input,
   and when the model uses it, a new tool span appears. No tracing code
   changed.

6. **Record something extra.** In `tool_span()` in `tracing.py`, add
   `span.set_attribute("tool.argument_count", ...)` or any attribute you
   like. Find it on the tool spans in Phoenix. Then notice that `llm.py`
   didn't change.

7. **Compare models.** Set `LLM_MODEL` in `.env` to another model loaded
   in LM Studio, run the same question, and compare the two traces: number of
   turns, tokens per call and time per call.

## 11. Troubleshooting

**No new traces in either tool.**
- Is Docker running? `colima status`, then `docker compose ps`. After a
  reboot, run `colima start`; the containers start by themselves.
- Is `setup_tracing()` called in `main.py` before `call_llm()`?

**Traces in Phoenix but not Langfuse.**
- Langfuse takes a few seconds to process new traces (see section 9).
  Refresh after a moment.
- Is `LANGFUSE_PUBLIC_KEY` set in `.env`? Without it, Langfuse is skipped.
- Do `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` match
  `LANGFUSE_INIT_PROJECT_PUBLIC_KEY` and `LANGFUSE_INIT_PROJECT_SECRET_KEY`?
  The init keys are only applied the first time Langfuse starts, so changing
  them later doesn't change the existing project. To start over, run
  `docker compose down -v` (this deletes all traces) and then
  `docker compose up -d`.
- Check the logs: `docker compose logs langfuse-web --tail 50`.

**The program prints "Transient error ..." and "Failed to export spans
batch".** The exporter couldn't reach a tool. It retries for a while first, so
the program may pause before exiting. The agent's own work still completes;
only the spans for that tool are lost. Start the containers and run again.

**Langfuse login doesn't work.** The email and password are
`LANGFUSE_INIT_USER_EMAIL` and `LANGFUSE_INIT_USER_PASSWORD` in `.env`. Like
the keys, they're only applied on first start.

**Port already in use** when starting the containers. Something else is using
6006 or 3000, such as a Phoenix server started with `uvx`. Stop it, then run
`docker compose up -d` again.

## 12. Glossary

| Term | Meaning |
|---|---|
| **Trace** | Everything recorded for one run, as a tree of spans. |
| **Span** | One step: name, start and end time, attributes, status, parent. |
| **Root span** | The span with no parent. Here, `agent`. |
| **Attribute** | A key/value detail on a span, such as `input.value`. |
| **Span kind** | What sort of step a span is: AGENT, CHAIN, LLM, TOOL. |
| **Current span** | The span that new spans become children of. Set by `start_as_current_span()`. |
| **Instrumentation** | Code that creates spans. *Auto*-instrumentation does it for a library without changes to your code. |
| **OpenTelemetry (OTel)** | The open standard and libraries for creating and sending traces. |
| **OTLP** | The OpenTelemetry Protocol, the format traces are sent in. |
| **Tracer provider** | The OpenTelemetry object that receives every finished span and passes it to its processors. |
| **Span processor** | Collects finished spans and hands them to an exporter. |
| **Exporter** | Sends spans to one tool. |
| **Resource** | Details about the program sending the spans, such as the project name. |
| **OpenInference** | A set of attribute names and instrumentors for AI applications, created by Arize (the makers of Phoenix). |
| **Generation** | Langfuse's name for a model call (an LLM span). |
