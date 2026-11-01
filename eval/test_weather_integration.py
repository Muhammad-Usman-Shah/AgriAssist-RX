
"""
AgriAssist-RX
Phase 6 - Weather Integration Test

Tests the full LangGraph weather route.
Does not modify the existing agent.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_csv import app


def main():

    question = "What is the weather in Lahore today?"

    print("=" * 60)
    print("AGRIASSIST-RX WEATHER INTEGRATION TEST")
    print("=" * 60)

    print("\nQuestion:", question)

    state = {
        "question": question,
        "selected_tool": "",
        "calculation": "none",
        "arg1": None,
        "arg2": None,
        "answer": "",
        "sources": [],
        "supported": False,
        "image_path": "",
        "vision_prediction": {},
        "api_error": "",
    }

    try:
        result = app.invoke(state)

        selected_tool = result.get("selected_tool")
        answer = str(result.get("answer", "")).strip()

        print("\nSelected tool:", selected_tool)
        print("Answer:", answer)
        print("Supported:", result.get("supported"))

        if selected_tool != "weather":
            print("\nRESULT: FAIL")
            print("The agent did not select the weather tool.")

        elif not answer:
            print("\nRESULT: FAIL")
            print("The weather tool returned an empty answer.")

        elif any(
            term in answer.lower()
            for term in [
                "unavailable",
                "error",
                "unable to",
                "failed",
                "try again",
            ]
        ):
            print("\nRESULT: REVIEW REQUIRED")
            print(
                "Weather routing succeeded, but the "
                "response may indicate a service failure."
            )

        else:
            print("\nRESULT: PASS")
            print(
                "Weather tool selected and "
                "a non-empty response returned."
            )
            print(
                "Weather data accuracy has not "
                "been independently verified."
            )

    except Exception as error:
        print("\nRESULT: ERROR")
        print(type(error).__name__, str(error))


if __name__ == "__main__":
    main()
