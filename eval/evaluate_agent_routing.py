
"""
AgriAssist-RX
Phase 6 - Step 1

Automated evaluation of six-route
LangGraph agent routing.

This checks whether the router selects
the expected tool for each question.

It does NOT evaluate answer correctness,
weather accuracy, or image predictions.
"""

import csv
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

# -------------------------------------------------
# Locate project folders
# -------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

from agent_csv import choose_tool


# -------------------------------------------------
# Evaluation dataset - 30 questions
# -------------------------------------------------

TESTS = [
    # RAG: 5 questions
    {
        "id": "R01",
        "question": "What are the symptoms of wheat stem rust?",
        "expected": "rag",
    },
    {
        "id": "R02",
        "question": "What causes wheat leaf rust?",
        "expected": "rag",
    },
    {
        "id": "R03",
        "question": "How does wheat stripe rust spread?",
        "expected": "rag",
    },
    {
        "id": "R04",
        "question": "What are the signs of wheat rust infection?",
        "expected": "rag",
    },
    {
        "id": "R05",
        "question": "How can farmers recognize wheat stem rust?",
        "expected": "rag",
    },

    # Weather: 5 questions
    {
        "id": "W01",
        "question": "What is the current weather in Lahore?",
        "expected": "weather",
    },
    {
        "id": "W02",
        "question": "What is the temperature in Multan today?",
        "expected": "weather",
    },
    {
        "id": "W03",
        "question": "What is the weather forecast for Islamabad?",
        "expected": "weather",
    },
    {
        "id": "W04",
        "question": "What is the humidity in Faisalabad today?",
        "expected": "weather",
    },
    {
        "id": "W05",
        "question": "What is the wind speed in Karachi today?",
        "expected": "weather",
    },

    # Calculator: 5 questions
    {
        "id": "C01",
        "question": (
            "How many fertilizer bags are needed "
            "for 5 acres at 2 bags per acre?"
        ),
        "expected": "calculator",
    },
    {
        "id": "C02",
        "question": (
            "Calculate the total seed needed "
            "for 4 acres at 50 kg per acre."
        ),
        "expected": "calculator",
    },
    {
        "id": "C03",
        "question": (
            "Calculate irrigation water for "
            "3 acres at 2000 liters per acre."
        ),
        "expected": "calculator",
    },
    {
        "id": "C04",
        "question": (
            "Calculate fertilizer cost for "
            "6 bags at 4500 PKR per bag."
        ),
        "expected": "calculator",
    },
    {
        "id": "C05",
        "question": (
            "If wheat yield is 1800 kg per acre, "
            "calculate production from 3 acres."
        ),
        "expected": "calculator",
    },

    # CSV: 5 questions
    {
        "id": "D01",
        "question": (
            "What is the average yield "
            "in crop_production.csv?"
        ),
        "expected": "csv",
    },
    {
        "id": "D02",
        "question": (
            "How many rows are in "
            "crop_production.csv?"
        ),
        "expected": "csv",
    },
    {
        "id": "D03",
        "question": (
            "What is the total fertilizer cost "
            "in crop_production.csv?"
        ),
        "expected": "csv",
    },
    {
        "id": "D04",
        "question": (
            "What is the minimum yield "
            "in the CSV file crop_production.csv?"
        ),
        "expected": "csv",
    },
    {
        "id": "D05",
        "question": (
            "What is the maximum area "
            "in crop_production.csv?"
        ),
        "expected": "csv",
    },

    # Vision: 5 questions
    {
        "id": "V01",
        "question": (
            "Analyze this image: "
            r"D:\AgriAssist-Datasets\train"
            r"\Aphid\aphid_27.png"
        ),
        "expected": "vision",
    },
    {
        "id": "V02",
        "question": "Analyze this wheat image.",
        "expected": "vision",
    },
    {
        "id": "V03",
        "question": "Classify this photo.",
        "expected": "vision",
    },
    {
        "id": "V04",
        "question": (
            "Identify disease from photo."
        ),
        "expected": "vision",
    },
    {
        "id": "V05",
        "question": "Predict from image.",
        "expected": "vision",
    },

    # Unsupported: 5 questions
    {
        "id": "U01",
        "question": (
            "Who won the FIFA World Cup in 2018?"
        ),
        "expected": "unsupported",
    },
    {
        "id": "U02",
        "question": "Write a romantic poem.",
        "expected": "unsupported",
    },
    {
        "id": "U03",
        "question": (
            "What is the capital of France?"
        ),
        "expected": "unsupported",
    },
    {
        "id": "U04",
        "question": (
            "Explain how to build a gaming PC."
        ),
        "expected": "unsupported",
    },
    {
        "id": "U05",
        "question": (
            "Recommend a science fiction movie."
        ),
        "expected": "unsupported",
    },
]


# -------------------------------------------------
# Prepare a state for the router
# -------------------------------------------------

def make_state(question):

    return {
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
    }


# -------------------------------------------------
# Run evaluation
# -------------------------------------------------

def main():

    results = []

    print("=" * 55)
    print("AGRIASSIST-RX: PHASE 6 ROUTING EVALUATION")
    print("=" * 55)

    for test in TESTS:

        state = make_state(test["question"])

        try:

            output = choose_tool(state)

            actual = output.get(
                "selected_tool", "missing"
            )

            status = (
                "PASS"
                if actual == test["expected"]
                else "FAIL"
            )

            detail = output.get("answer", "")

            # Keep API outages separate from wrong routing.
            if (
                actual == "unsupported"
                and test["expected"] != "unsupported"
                and (
                    "503" in detail
                    or "UNAVAILABLE" in detail
                    or "router is temporarily unavailable"
                    in detail.lower()
                )
            ):
                status = "ERROR"

        except Exception as error:

            actual = "error"
            status = "ERROR"
            detail = str(error)

        record = {
            "id": test["id"],
            "question": test["question"],
            "expected": test["expected"],
            "actual": actual,
            "status": status,
            "details": detail,
        }

        results.append(record)

        print(
            f"{test['id']} | "
            f"Expected: {test['expected']:<12} | "
            f"Actual: {actual:<12} | "
            f"{status}"
        )

    # ---------------------------------------------
    # Calculate results
    # ---------------------------------------------

    passed = sum(
        r["status"] == "PASS"
        for r in results
    )

    failed = sum(
        r["status"] == "FAIL"
        for r in results
    )

    errors = sum(
        r["status"] == "ERROR"
        for r in results
    )

    total = len(results)
    completed = passed + failed

    accuracy = (
        100 * passed / completed
        if completed
        else None
    )

    # ---------------------------------------------
    # Save JSON and CSV reports
    # ---------------------------------------------

    output_dir = ROOT / "eval" / "results"
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    summary = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "total_cases": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "accuracy_percent": accuracy,
        "note": (
            "Routing-only evaluation. "
            "Answer quality not evaluated. "
            "Accuracy excludes ERROR cases."
        ),
        "results": results,
    }

    json_path = (
        output_dir / "routing_evaluation.json"
    )

    csv_path = (
        output_dir / "routing_evaluation.csv"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "question",
                "expected",
                "actual",
                "status",
                "details",
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 55)
    print("EVALUATION SUMMARY")
    print("=" * 55)

    print(f"Total questions: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"API/runtime errors: {errors}")

    if accuracy is not None:
        print(
            f"Routing accuracy (completed cases): "
            f"{accuracy:.2f}%"
        )
    else:
        print(
            "Routing accuracy: unavailable"
        )

    print("\nReports saved:")
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
