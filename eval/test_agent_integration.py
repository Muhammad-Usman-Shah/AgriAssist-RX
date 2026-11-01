
"""
AgriAssist-RX
Phase 6 - Integrated Agent Evaluation

Initial smoke test:
Verify that the LangGraph agent can execute
a RAG question end to end.

Does not modify the existing agent.
"""

import json
import sys

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_csv import app


def main():

    question = (
        "What are the symptoms of wheat stem rust?"
    )

    print("=" * 60)
    print("AGRIASSIST-RX INTEGRATION SMOKE TEST")
    print("=" * 60)

    print("\nQuestion:", question)

    initial_state = {
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

        result = app.invoke(initial_state)

        print("\nSelected tool:",
              result.get("selected_tool"))

        print("\nAnswer:")
        print(result.get("answer"))

        print("\nSources:")
        print(
            json.dumps(
                result.get("sources", []),
                indent=2,
                ensure_ascii=False,
                default=str
            )
        )

        selected_tool = result.get(
            "selected_tool"
        )

        answer = str(
            result.get("answer", "")
        ).strip()

        if (
            selected_tool == "rag"
            and answer
            and result.get("supported") is True
            and result.get("sources")
        ):

            print(
                "\nRESULT: PASS"
                "\nRAG route executed with citations."
            )

        else:

            print(
                "\nRESULT: REVIEW REQUIRED"
                "\nThe expected RAG behavior was "
                "not fully confirmed."
            )

    except Exception as error:

        print("\nRESULT: ERROR")
        print(type(error).__name__, str(error))


if __name__ == "__main__":
    main()
