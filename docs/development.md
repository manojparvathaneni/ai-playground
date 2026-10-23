# Working on the playground

How this project is organised and the habits that keep it working as it
grows. They're the same practices production teams use, at a size that fits
a learning project.

## Contents

1. [Where code goes](#1-where-code-goes)
2. [The rules for each place](#2-the-rules-for-each-place)
3. [The everyday routine](#3-the-everyday-routine)
4. [The checks, and what each one catches](#4-the-checks-and-what-each-one-catches)
5. [Settings and secrets](#5-settings-and-secrets)
6. [Dependencies](#6-dependencies)
7. [Evaluations](#7-evaluations)
8. [Adding an experiment, step by step](#8-adding-an-experiment-step-by-step)
9. [Keeping things up to date](#9-keeping-things-up-to-date)

---

## 1. Where code goes

| Place | What belongs there |
|---|---|
| `src/ai_playground/` | Code more than one experiment uses: the agent loop, tools, tracing, settings, evaluations. |
| `experiments/NN_topic/` | One idea being explored: a script, its notes, anything specific to it. |
| `tests/` | Tests for `src/ai_playground/`. |
| `docs/` | Guides longer than a README. |

**Start in an experiment; move code to `src/` when a second experiment needs
it.** Copying a function once is fine. The second copy is the signal to move
it into `src/`, add type hints and tests, and import it from both places.
Moving code too early creates shared code nobody else uses; moving it too
late leaves several slightly different copies.

Production projects follow the same split, under different names: a
*library* of reusable code, and *applications* that use it.

## 2. The rules for each place

**In `src/ai_playground/`:**

- **Type hints on everything.** Ruff fails if a function is missing them,
  and ty checks that the code matches them. They document what each function
  takes and returns, and catch mistakes before the code runs.
- **Tests for everything.** Put them in `tests/test_<module>.py`. They must
  run offline: use the fakes in `tests/fakes.py` instead of a real model.
- **Settings from `Settings`**, never hardcoded (see section 5).
- **Spans through the helpers in `tracing.py`**, never by setting attribute
  names directly.
- **A docstring on every module and public function**, saying what it's for.

**In `experiments/`:**

- **Clarity first.** Type hints and strictness are optional; experiments
  are where you try things out.
- **A `README.md`** in each folder: what it explores, how to run it, which
  dependency group it needs, and what you found.
- **Tests are optional but welcome**, as `test_*.py` next to the code (see
  `02_evaluation/test_time_grader.py`). `make test` finds them.

## 3. The everyday routine

```sh
git switch -c my-change          # 1. start a branch for the change
# ... edit code ...
make format                      # 2. fix formatting automatically
make check                       # 3. run every check CI runs
git add -A && git commit         # 4. commit with a message saying why
git push -u origin my-change     # 5. push, and open a pull request
```

CI runs `make check` again on the pull request. Merge once it passes.

**Why branches?** `main` should always work. A branch lets you try something,
commit as often as you like, and throw it away if it doesn't work out,
without breaking `main`.

**Why small commits?** Each commit should do one thing, with a message that
says what and why. When something breaks later, `git log` and `git bisect`
can find the commit that caused it, and a small commit is easy to
understand or undo.

## 4. The checks, and what each one catches

`make check` runs them all. You can also run each one on its own:

| Command | Tool | Catches |
|---|---|---|
| `make lint` | ruff | Likely bugs (unused variables, undefined names, risky patterns), messy imports, missing type hints in `src/`, and code that isn't formatted. |
| `make typecheck` | ty | Calling a function with the wrong type, forgetting that something can be `None`, typos in attribute names. |
| `make test` | pytest | Code that doesn't do what it should. |
| `make lock` | uv | `uv.lock` or `requirements.txt` out of step with `pyproject.toml`. |

`make format` fixes formatting and anything else ruff can fix by itself.

The same checks run in GitHub Actions
([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)) on every push to
`main` and every pull request.

## 5. Settings and secrets

Every setting is a field on `Settings` in
[`config.py`](../src/ai_playground/config.py), with a default. Change one by
setting an environment variable of the same name in capitals, or by adding a
line to `.env`:

```sh
LLM_MODEL=qwen/qwen3.6-35b-a3b uv run python experiments/02_evaluation/main.py
```

To add a setting, add a field with a default to `Settings`, and a commented
line to `.env.example` so others know it exists.

**Secrets** (API keys, passwords) go only in `.env`, which git ignores. Never
put them in code or in `.env.example`. If one is committed by accident,
removing it in a later commit isn't enough: it's still in the history, so
change the secret itself.

## 6. Dependencies

`pyproject.toml` lists what the project needs; `uv.lock` records the exact
versions, so every machine installs the same thing. Commit both.

| Kind | Add with | Examples |
|---|---|---|
| Every experiment needs it | `uv add <package>` | openai, opentelemetry |
| Only some experiments need it | `uv add --group <name> <package>` | a vector database for RAG |
| Development tools | `uv add --dev <package>` | pytest, ruff, ty |

Install a group with `uv sync --group <name>`, and say which group an
experiment needs in its README. After any change, run `make requirements` to
regenerate `requirements.txt`; `make check` fails until you do.

## 7. Evaluations

Tests check the code; evaluations check the model's behaviour (see
[experiment 02](../experiments/02_evaluation/)). **Run the evaluation before
merging any change that could change what the agent does**: a prompt, a tool
description, the model, the agent loop. Compare the pass rate with the last
run, and look at any new failures in Phoenix or Langfuse.

Evaluations need a model server, so they don't run in CI. A production team
would run them on a schedule or before each release, against a hosted model.

## 8. Adding an experiment, step by step

1. `git switch -c experiment-03-<topic>`
2. Create `experiments/03_<topic>/` with `main.py` and `README.md`. Copy the
   shape of experiment 01: call `setup_tracing()` first, then use
   `ai_playground`.
3. If it needs new packages, add them to a group:
   `uv add --group <topic> <package>`, then `make requirements`.
4. Run it, look at the traces, and write what you found in its README.
5. Add a row to the experiments table in the main [README](../README.md).
6. `make check`, commit, push, pull request.

## 9. Keeping things up to date

| What | How |
|---|---|
| Python packages | `uv lock --upgrade`, then `make requirements` and `make check` |
| GitHub Actions | Dependabot ([`.github/dependabot.yml`](../.github/dependabot.yml)) opens a pull request each week a new version is out; merge it once CI passes |
| Phoenix | Change the image tag in [`docker-compose.yml`](../docker-compose.yml) (pinned to a major version, such as `version-20`), then `docker compose pull && make up` |
| Langfuse | Replace [`docker/langfuse/docker-compose.yml`](../docker/langfuse/docker-compose.yml) with the latest official file, keeping the header, then `docker compose pull && make up` |

Upgrade on a branch, and run `make check` and the evaluation before merging,
so a breaking change in a new version shows up there and not in `main`.
