# AI playground

A place to learn to build AI applications, from the basics to advanced, one
experiment at a time. Every run is traced, so you can see exactly what the
model and your code did.

## Layout

```
├── src/ai_playground/      shared code that experiments reuse
│   ├── config.py           Settings: every setting, read from env vars and .env
│   ├── llm.py              call_llm(): the agent loop
│   ├── tools.py            example tools, and make_tool_schema()
│   ├── tracing.py          setup_tracing(), span helpers, sessions, experiment labels
│   ├── evals.py            run cases through an application and grade them
│   ├── scores.py           send eval scores to Phoenix and Langfuse
│   ├── datasets.py         keep eval cases as datasets, and runs as experiments
│   └── http_json.py        small helpers for the tools' JSON APIs
├── experiments/            one numbered folder per experiment
├── tests/                  offline tests for src/ai_playground
├── docs/                   longer guides
├── docker/ + docker-compose.yml   Phoenix and Langfuse, for tracing
├── .github/workflows/      CI: runs `make check` on every push and pull request
├── Makefile                shortcuts: `make check`, `make up`, ... (run `make` to list)
├── pyproject.toml          dependencies and tool settings
└── .env.example            template for .env (settings and secrets)
```

**Shared code vs experiments.** Start new ideas in an experiment. Move code
into `src/ai_playground/` when a second experiment needs it; there it gets
type hints and tests. See [docs/development.md](docs/development.md) for the
rules and the everyday routine.

## Getting started

You need [uv](https://docs.astral.sh/uv/), and a local model server that
supports the OpenAI Responses API and function calling, such as
[LM Studio](https://lmstudio.ai) on `http://localhost:1234`.

```sh
make install                                     # create .venv with all dependencies
cp .env.example .env                             # then fill in the secrets (see Tracing)
make up                                          # start Phoenix and Langfuse
uv run python experiments/01_tool_calling/main.py
```

The model, server and other settings have defaults in
[`config.py`](src/ai_playground/config.py); override any of them in `.env`,
for example `LLM_MODEL=qwen/qwen3.6-35b-a3b`.

## Experiments

| # | Experiment | What it shows |
|---|---|---|
| 01 | [Tool calling](experiments/01_tool_calling/) | The agent loop: the model asks for tools, we run them, it answers. Traced in Phoenix and Langfuse. |
| 02 | [Evaluation](experiments/02_evaluation/) | Scoring the agent on a set of questions, with the scores shown in Phoenix and Langfuse. |
| 03 | [Pydantic AI](experiments/03_pydantic_ai/) | Structured output: Pydantic models check the model's reply, and a Pydantic AI agent retries with the errors. Traced with the framework's own spans. |

To add one, see [Adding an experiment](docs/development.md#8-adding-an-experiment-step-by-step).
Packages that only some experiments need go in a dependency group, so the
base environment stays small; see [Dependencies](docs/development.md#6-dependencies).

## Guides

- [docs/development.md](docs/development.md): where code goes, the checks,
  settings, dependencies, and the everyday routine.
- [docs/tracing.md](docs/tracing.md): traces and spans, how they reach
  Phoenix and Langfuse, scores, and exercises.
- [docs/pydantic.md](docs/pydantic.md): Pydantic and Pydantic AI for model
  output, with examples and exercises. Read it before experiment 03.

## Tracing

Each run is traced and sent to two observability tools, so you can compare
the same run side by side: [Arize Phoenix](https://arize.com/docs/phoenix)
and [Langfuse](https://langfuse.com). Both run locally in Docker.

```
agent
├── turn 1
│   ├── Response                      (model call, traced automatically)
│   ├── get_current_time              (tool)
│   └── get_current_time_in_timezone  (tool)
└── turn 2
    └── Response                      (final answer)
```

### One-time setup

1. Install a Docker runtime. On macOS, [Colima](https://github.com/abiosoft/colima)
   is free and command-line only:

   ```sh
   brew install colima docker docker-compose
   colima start --cpu 4 --memory 8
   ```

   For `docker compose` to work, `~/.docker/config.json` must contain
   `"cliPluginsExtraDirs": ["/opt/homebrew/lib/docker/cli-plugins"]`.

2. Create `.env` from `.env.example` and replace every `CHANGEME` with a
   secret from `openssl rand -hex 32`. Values that must match are noted in
   the file. `.env` is ignored by git.

### Each time

1. Start the tools (Colima first, if it was stopped):

   ```sh
   colima start
   docker compose up -d
   ```

2. Run an experiment.

3. Open both and pick the `agentic-ai-course` project:
   - Phoenix: <http://localhost:6006>
   - Langfuse: <http://localhost:3000>, logging in with
     `LANGFUSE_INIT_USER_EMAIL` and `LANGFUSE_INIT_USER_PASSWORD` from `.env`

To stop: `docker compose down` (traces are kept), or `docker compose down -v`
to delete them too. Without the `LANGFUSE_*` keys in `.env`, traces go to
Phoenix only.

To keep your traces and scores when moving to another machine, use
`make backup` and `make restore`; see
[where the data lives](docs/tracing.md#where-the-data-lives-and-moving-it-to-another-machine).

### How it fits together

All tracing code lives in `src/ai_playground/tracing.py`:

- `setup_tracing()` creates one OpenTelemetry exporter per tool, so every
  span is sent to each of them, and traces OpenAI calls automatically.
- `agent_span()`, `turn_span()` and `tool_span()` create our own spans with
  OpenInference attribute names, which both Phoenix and Langfuse understand.
  `llm.py` uses these helpers and never sets attributes itself.

If `setup_tracing()` is not called, the spans do nothing.

**To learn more**, read [docs/tracing.md](docs/tracing.md). It explains traces
and spans, how the code builds the tree above, how spans reach both tools,
what each Docker container does, and has exercises and troubleshooting tips.

## Checks

```sh
make check      # lint, formatting, type checks, tests, lock file: what CI runs
make format     # fix formatting automatically
```

The tests run offline, with no model server or Docker needed. See
[the checks](docs/development.md#4-the-checks-and-what-each-one-catches) for
what each one catches.
