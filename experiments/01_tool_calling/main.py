"""Experiment 01: a model that calls Python tools to answer a question."""

from dotenv import load_dotenv

from ai_playground.llm import call_llm
from ai_playground.tools import get_current_time, get_current_time_in_timezone
from ai_playground.tracing import setup_tracing

def main():
    # Read settings such as the Langfuse keys from the .env file.
    load_dotenv()

    # Start tracing first, so every step below is recorded.
    setup_tracing()

    print("Starting the program...\n")
    response = call_llm(
        "What is the current local time and the current time in UTC?",
        model="google/gemma-4-e4b",
        tools=[get_current_time, get_current_time_in_timezone],
    )
    print(f"LLM response: {response.output_text}")

if __name__ == "__main__":
    main()