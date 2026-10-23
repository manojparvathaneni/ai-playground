# 01 · Tool calling

A model answers *"What is the current local time and the current time in
UTC?"* by calling two Python tools, `get_current_time` and
`get_current_time_in_timezone`, and then writing an answer from their results.

## What it shows

- **The agent loop** (`call_llm()` in
  [`llm.py`](../../src/ai_playground/llm.py)): send the conversation to the
  model, run any tools it asks for, add their results to the conversation,
  and repeat until the model answers without asking for more.
- **Describing tools to the model** (`make_tool_schema()` in
  [`tools.py`](../../src/ai_playground/tools.py)): a function's name,
  docstring and type hints become the description the model reads. The file
  also shows the hand-written version, for comparison.
- **Tracing**: each turn and tool call appears as a span in Phoenix and
  Langfuse. See [docs/tracing.md](../../docs/tracing.md).

## Run it

Needs a model server on `http://localhost:1234` (such as LM Studio) with
`google/gemma-4-e4b` loaded, or set `LLM_MODEL` in `.env`. For traces,
start the containers first (`docker compose up -d`).

```sh
uv run python experiments/01_tool_calling/main.py
```

No extra dependency group is needed.

## Try next

The exercises in [docs/tracing.md](../../docs/tracing.md#10-exercises) build on
this experiment: cause an error, change the number of turns, add a tool, or
compare models.
