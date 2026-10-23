"""Experiment 01: a model that calls Python tools to answer a question."""

from ai_playground.llm import call_llm
from ai_playground.tools import get_current_time, get_current_time_in_timezone
from ai_playground.tracing import setup_tracing


def main():
    # Start tracing first, so every step below is recorded.
    # Settings (model, server, Langfuse keys) come from .env.
    setup_tracing()

    print("Starting the program...\n")
    response = call_llm(
        "What is the current local time and the current time in UTC?",
        tools=[get_current_time, get_current_time_in_timezone],
    )
    print(f"LLM response: {response.output_text}")


if __name__ == "__main__":
    main()
