
"""
AgriAssist-RX
Phase 6 - Unsupported Request Integration Test

Checks whether the complete LangGraph agent:
1. Recognizes an unsupported request.
2. Selects the unsupported route.
3. Returns an appropriate response.

Does not modify the existing agent.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_csv import app


def main():

    question = (
        "Who won the FIFA World Cup in 2018?"
    )

    print("=" * 60)
    print("AGRIASSIST-RX UNSUPPORTED INTEGRATION TEST")
    print("=" * 60)

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
        answer = str(
            result.get("answer", "")
        ).strip()

        print("\nSelected tool:", selected_tool)
        print("Answer:", answer)
        print("Supported:", result.get("supported"))

        if selected_tool == "api_error":
            print("\nRESULT: NOT EVALUATED")
            print("AI routing service was unavailable.")

        elif selected_tool != "unsupported":
            print("\nRESULT: FAIL")
            print("Expected unsupported route.")

        elif not answer:
            print("\nRESULT: FAIL")
            print("No explanation was returned.")

        elif result.get("supported") is True:
            print("\nRESULT: REVIEW REQUIRED")
            print(
                "Unsupported requests should not "
                "be marked as evidence-supported."
            )

        else:
            print("\nRESULT: PASS")
            print(
                "Unsupported request was routed "
                "correctly and handled safely."
            )

    except Exception as exc:
        print("\nRESULT: ERROR")
        print(type(exc).__name__, str(exc))


if __name__ == "__main__":
    main()
