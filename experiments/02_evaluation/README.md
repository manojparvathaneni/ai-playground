# 02 · Evaluation

Measures how well the tool-calling agent from
[experiment 01](../01_tool_calling/) answers six questions about the time,
and records each verdict in Phoenix and Langfuse.

## Why evaluate?

Tests check code that always gives the same answer. A model doesn't: it
phrases things differently each time, and sometimes gets them wrong. So
instead of asserting one exact answer, an **evaluation** runs a set of
**cases** (questions) through the application and lets a **grader** decide
whether each answer is good enough. The result is a **pass rate**.

Run it after every change (a new model, prompt or tool) and compare the pass
rate with the last run. That's how you know a change made things better
rather than just different.

## How it works

| Piece | Where | What it does |
|---|---|---|
| Cases | `CASES` in [`main.py`](main.py) | The questions, and which timezones each answer must give the time for |
| Application | `answer()` in `main.py` | The agent from experiment 01 |
| Grader | [`time_grader.py`](time_grader.py) | Works out the correct time when it runs, and passes the answer if it states that time (within 2 minutes). Accepts "16:05" and "4:05 PM", but not a wrong AM/PM. |
| Runner | [`evals.py`](../../src/ai_playground/evals.py) | Runs and grades every case, traces both steps, prints a report |
| Scores | [`scores.py`](../../src/ai_playground/scores.py) | Attaches each verdict to its trace in Phoenix and Langfuse |

The runner and scores are in `src/` because every evaluation will use them.
The cases and the grader are here, because they're specific to this
application. A new evaluation needs only those two things.

The grader has its own tests in [`test_time_grader.py`](test_time_grader.py):
a grader with a bug gives misleading scores, so it deserves tests as much as
any other code.

## Run it

Start the model server and the containers (`make up`), then:

```sh
uv run python experiments/02_evaluation/main.py
```

It prints one line per case, details for each failure, and the pass rate. It
exits with an error if fewer than 80% of cases pass, so a script or a CI job
can use it as a gate.

To see the results in the tools, open the `agentic-ai-course` project:

- **Phoenix** (<http://localhost:6006>): each case is a trace named
  `eval: <case>`, with a `correct` annotation. Filter on it to see only the
  failures.
- **Langfuse** (<http://localhost:3000>): the same traces, each with a
  `correct` score.

Each run is one **session**, named like `02_evaluation-20261023-021411` and
printed at the end, so a run's six traces appear together on both tools'
Sessions pages. Every trace is also labelled with the model and git commit;
see [sessions](../../docs/tracing.md#sessions-the-traces-of-one-run).

## What it found

In the first runs, 5 of 6 cases passed. The `local` case ("What's the time
here?") failed: instead of calling `get_current_time`, the model asked what
"here" meant. That's a real weakness of the agent, found by the evaluation,
not a bug in the code.

Asking that one question 10 times showed it was worse than one run
suggested: only 2 of 10 answers called the tool. Two changes in
[`tools.py`](../../src/ai_playground/tools.py) fixed it:

1. **The tool's description.** `get_current_time` said only "Get the current
   local time." It now says it gives the time where the user is, and to use
   it for "here" or when no place is named. The `local` question went from
   2/10 to 10/10.
2. **The tool's output.** The full evaluation then showed a second failure,
   seen at 1 AM: the tool returned `01:13:00` and the model said "1:13 PM".
   The tools now return both clocks, `01:13:00 (1:13 AM)`, so there is
   nothing to guess.

After both changes, three runs in a row passed 6/6. The lesson: the model
only knows what the tool's description and output tell it, so they are part
of the prompt.

## Try next

- **Compare models.** Set `LLM_MODEL` in `.env` to another model and compare
  pass rates.
- **Add cases.** Add questions the agent might get wrong, such as a city with
  a half-hour offset (India) or a question with no time in it at all.
- **Run it several times.** Models aren't deterministic, so one run can pass
  by luck. Run it three times and see whether the same cases fail.
