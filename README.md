# AI playground

A place to learn to build AI applications, from the basics to advanced, one
experiment at a time. Every run is traced, so you can see exactly what the
model and your code did.

## Layout

```
├── src/ai_playground/      shared code that experiments reuse
│   ├── llm.py              call_llm(): the agent loop
│   ├── tools.py            example tools, and make_tool_schema()
│   └── tracing.py          setup_tracing() and span helpers
├── experiments/            one numbered folder per experiment
│   └── 01_tool_calling/    a model that calls Python tools
├── tests/                  offline tests for src/ai_playground
├── docs/                   longer guides, such as tracing.md
├── docker/                 container setup that needs its own folder
├── docker-compose.yml      Phoenix and Langfuse, for tracing
├── pyproject.toml          project settings and dependencies
└── .env.example            template for .env (settings and secrets)
```

**Shared code vs experiments.** Code goes in `src/ai_playground/` once more
than one experiment needs it, or once it's worth testing. Everything else
stays in the experiment's folder. Start new ideas in an experiment, and move
pieces into `src/` when you find yourself copying them.

## Getting started

You need [uv](https://docs.astral.sh/uv/), and a local model server that
supports the OpenAI Responses API and function calling, such as
[LM Studio](https://lmstudio.ai) on `http://localhost:1234`.

```sh
uv sync                                          # create .venv with the dependencies
uv run python experiments/01_tool_calling/main.py
```

Tracing needs Phoenix and Langfuse running; see [Tracing](#tracing) below.

## Experiments

| # | Experiment | What it shows |
|---|---|---|
| 01 | [Tool calling](experiments/01_tool_calling/) | The agent loop: the model asks for tools, we run them, it answers. Traced in Phoenix and Langfuse. |

### Adding an experiment

1. Create the next numbered folder, such as `experiments/02_<topic>/`, with a
   `main.py` and a short `README.md` saying what it explores.
2. Import shared code from `ai_playground`, for example
   `from ai_playground.llm import call_llm`.
3. Run it with `uv run python experiments/02_<topic>/main.py`.
4. Add it to the table above.

### Adding packages

Packages that every experiment uses are in `dependencies` in
`pyproject.toml`; keep that list short. Packages only some experiments need
go in a named **dependency group**, so the base environment stays small:

```sh
uv add --group rag chromadb     # add chromadb to a "rag" group
uv sync --group rag             # install the base packages plus that group
```

The experiment's README should say which group it needs. `uv.lock` records
the exact versions; commit it along with `pyproject.toml`.
`requirements.txt` is generated from `uv.lock` for tools that need one; see
its header for how to regenerate it.

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

## Tests

The tests cover the shared code and run offline, with no model or Docker
needed:

```sh
uv run python -m unittest discover -s tests
```
