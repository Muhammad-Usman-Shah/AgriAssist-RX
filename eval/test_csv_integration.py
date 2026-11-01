
"""
AgriAssist-RX
Phase 6 - CSV Integration Test

Tests the full LangGraph CSV analysis route.
Does not modify the existing agent.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_csv import app


def main():

    question = (
        "Analyze crop_production.csv and "
        "calculate the average crop yield."
    )

    print("=" * 60)
    print("AGRIASSIST-RX CSV INTEGRATION TEST")
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
        answer = str(
            result.get("answer", "")
        ).strip()

        print("\nSelected tool:", selected_tool)
        print("Answer:", answer)
        print("Supported:", result.get("supported"))

        if selected_tool != "csv":
            print("\nRESULT: FAIL")
            print("CSV tool was not selected.")

        elif not answer:
            print("\nRESULT: FAIL")
            print("CSV tool returned an empty answer.")

        elif any(
            term in answer.lower()
            for term in [
                "file not found",
                "unable to",
                "failed to",
                "error reading",
            ]
        ):
            print("\nRESULT: REVIEW REQUIRED")
            print(
                "CSV routing worked, but "
                "the tool may have failed."
            )

        else:
            print("\nRESULT: PASS")
            print(
                "CSV route executed and "
                "returned a non-empty answer."
            )
            print(
                "Numerical accuracy still "
                "requires verification."
            )

    except Exception as error:

        print("\nRESULT: ERROR")
        print(type(error).__name__, str(error))


if __name__ == "__main__":
    main()
