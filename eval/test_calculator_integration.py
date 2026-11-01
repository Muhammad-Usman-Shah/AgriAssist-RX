
"""
AgriAssist-RX
Phase 6 - Calculator Integration Test

Tests whether the full LangGraph agent:
1. Recognizes a calculation request.
2. Selects the calculator tool.
3. Returns the correct numerical result.

Does not modify the existing agent.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_csv import app


def main():

    question = (
        "If wheat yield is 1800 kg per acre, "
        "calculate production from 3 acres."
    )

    print("=" * 60)
    print("AGRIASSIST-RX CALCULATOR INTEGRATION TEST")
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

        selected_tool = result.get(
            "selected_tool"
        )

        answer = str(
            result.get("answer", "")
        )

        print("\nSelected tool:", selected_tool)
        print("Calculation:", result.get("calculation"))
        print("Answer:", answer)

        expected_value = 5400

        normalized_answer = answer.replace(",", "")

        if (
            selected_tool == "calculator"
            and str(expected_value) in normalized_answer
        ):

            print("\nRESULT: PASS")
            print(
                "Calculator route executed "
                "and returned the expected value."
            )

        else:

            print("\nRESULT: REVIEW REQUIRED")
            print(
                "Expected the calculator route "
                "and a production value of 5400 kg."
            )

    except Exception as error:

        print("\nRESULT: ERROR")
        print(type(error).__name__, str(error))


if __name__ == "__main__":
    main()
