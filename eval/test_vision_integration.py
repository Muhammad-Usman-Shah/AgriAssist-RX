
"""
AgriAssist-RX
Phase 6 - Vision Integration Test

Tests whether the complete LangGraph agent:
1. Recognizes an image classification request.
2. Selects the vision tool.
3. Processes an existing image.
4. Returns a non-empty prediction.

This is an integration smoke test,
not an independent accuracy evaluation.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_csv import app


IMAGE_PATH = Path(
    r"D:\AgriAssist-Datasets\train\Aphid\aphid_27.png"
)


def main():

    print("=" * 60)
    print("AGRIASSIST-RX VISION INTEGRATION TEST")
    print("=" * 60)

    if not IMAGE_PATH.is_file():
        print("RESULT: NOT RUN")
        print("Image file not found:", IMAGE_PATH)
        print("Update IMAGE_PATH to an existing image.")
        return

    question = (
        "Analyze this wheat image and identify "
        f"the disease: {IMAGE_PATH}"
    )

    state = {
        "question": question,
        "selected_tool": "",
        "calculation": "none",
        "arg1": None,
        "arg2": None,
        "answer": "",
        "sources": [],
        "supported": False,
        "image_path": str(IMAGE_PATH),
        "vision_prediction": {},
        "api_error": "",
    }

    try:
        result = app.invoke(state)

        selected_tool = result.get("selected_tool")
        answer = str(result.get("answer", "")).strip()
        prediction = result.get("vision_prediction")

        print("\nSelected tool:", selected_tool)
        print("Answer:", answer)
        print("Vision prediction:", prediction)

        if selected_tool != "vision":
            print("\nRESULT: FAIL")
            print("Expected the vision route.")
            return

        if not answer:
            print("\nRESULT: FAIL")
            print("The vision tool returned no answer.")
            return

        error_phrases = [
            "file not found",
            "model not found",
            "unable to process",
            "failed to load",
            "traceback",
        ]

        if any(
            phrase in answer.lower()
            for phrase in error_phrases
        ):
            print("\nRESULT: REVIEW REQUIRED")
            print("The response may indicate an error.")
            return

        if not prediction:
            print("\nRESULT: REVIEW REQUIRED")
            print("Vision route worked, but prediction data is empty.")
            return

        print("\nRESULT: PASS")
        print("Vision tool executed and returned a prediction.")
        print(
            "Classification accuracy is not established "
            "by this smoke test."
        )

    except Exception as exc:
        print("\nRESULT: ERROR")
        print(type(exc).__name__, str(exc))


if __name__ == "__main__":
    main()
