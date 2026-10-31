
"""
AgriAssist-RX
Phase 6: New 60-Question Routing Evaluation

Purpose:
- Evaluate six agent routes.
- Use questions different from the original 30.
- Record incorrect routing and API errors.
- Save JSON and CSV results.

Important:
This tests ROUTING ONLY.
It does not execute the selected tools or
evaluate generated answer correctness.
"""

import csv
import json
import sys

from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict


# ===========================================
# 1. Project configuration
# ===========================================

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

from agent_csv import choose_tool


# ===========================================
# 2. NEW 60-QUESTION DATASET
# ===========================================

TEST_GROUPS = {

    "rag": [
        "Why do orange pustules appear on wheat leaves?",
        "How can wheat farmers recognize leaf rust?",
        "Explain how fungal spores infect wheat plants.",
        "What conditions encourage wheat stem rust infection?",
        "How can farmers distinguish leaf rust from stem rust?",
        "Describe the symptoms of stripe rust on wheat leaves.",
        "Why is early identification of wheat rust important?",
        "What happens when rust severely damages wheat stems?",
        "Explain how wheat rust affects crop growth.",
        "What is the difference between wheat rust and leaf blight?",
    ],

    "weather": [
        "Give me today's weather report for Multan.",
        "What is today's temperature in Lahore?",
        "Show the weather forecast for Bahawalpur.",
        "Check humidity in Islamabad today.",
        "What is the wind speed in Faisalabad right now?",
        "Show current weather conditions in Peshawar.",
        "What is the temperature in Hyderabad today?",
        "Check today's weather in Sialkot.",
        "What is the current humidity in Karachi?",
        "Give the latest weather forecast for Rawalpindi.",
    ],

    "calculator": [
        "How much seed is required for 6 acres at 40 kg per acre?",
        "Find fertilizer quantity for 8 acres at 3 bags per acre.",
        "Calculate the cost of 9 fertilizer bags at 4200 PKR each.",
        "Convert 12 acres into hectares.",
        "Calculate irrigation volume for 2 acres with 25 mm water depth.",
        "A farmer uses 2 fertilizer bags per acre on 7 acres. Calculate total bags.",
        "What is the seed requirement for 10 acres at 45 kg per acre?",
        "Calculate crop production for 4 acres yielding 1500 kg per acre.",
        "If yield is 2200 kg per acre, calculate the harvest from 5 acres.",
        "Find total fertilizer cost for 12 bags priced at 3800 PKR per bag.",
    ],

    "csv": [
        "Find the mean fertilizer cost in crop_production.csv.",
        "Count the records in crop_production.csv.",
        "Show the highest wheat yield in crop_production.csv.",
        "Find the lowest field area in crop_production.csv.",
        "Calculate total yield from crop_production.csv.",
        "What is the minimum fertilizer cost in crop_production.csv?",
        "Show average cultivated area in crop_production.csv.",
        "How many data rows does crop_production.csv contain?",
        "Find the maximum fertilizer spending in crop_production.csv.",
        "Show total production in crop_production.csv.",
    ],

    "vision": [
        "Please examine this wheat photograph.",
        "Can you classify the disease shown in a wheat image?",
        "Analyze this crop picture.",
        "Identify the condition visible in my wheat photo.",
        "Check this leaf photograph for disease.",
        "Predict the disease from this wheat picture.",
        "Please analyze this image of wheat.",
        "What disease is visible in this wheat image?",
        "Classify the wheat leaf in my photograph.",
        "Run image classification on a wheat plant photo.",
    ],

    "unsupported": [
        "Who wrote the novel Pride and Prejudice?",
        "How can I repair my laptop screen?",
        "Give me the latest football transfer news.",
        "Write a birthday message for my friend.",
        "Explain the rules of chess.",
        "Recommend five action movies.",
        "What is the population of Canada?",
        "Help me design a gaming logo.",
        "Tell me a joke about astronauts.",
        "What is the exchange rate from USD to EUR?",
    ],
}


# ===========================================
# 3. Build test cases
# ===========================================

PREFIXES = {
    "rag": "R",
    "weather": "W",
    "calculator": "C",
    "csv": "D",
    "vision": "V",
    "unsupported": "U",
}

TESTS = []

for route, questions in TEST_GROUPS.items():

    for index, question in enumerate(
        questions, start=1
    ):

        TESTS.append({
            "id": f"{PREFIXES[route]}{index:02d}",
            "question": question,
            "expected": route,
        })

assert len(TESTS) == 60

assert len({
    item["question"].lower()
    for item in TESTS
}) == 60


# ===========================================
# 4. Create the agent state
# ===========================================

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


# ===========================================
# 5. Detect router availability errors
# ===========================================

def is_router_error(answer):

    text = str(answer).lower()

    indicators = [
        "router is temporarily unavailable",
        "gemini api key is missing",
        "503 unavailable",
        "429 resource_exhausted",
        "resource_exhausted",
        "quota exceeded",
    ]

    return any(
        indicator in text
        for indicator in indicators
    )


# ===========================================
# 6. Evaluate routing
# ===========================================

def main():

    results = []

    print("=" * 65)
    print("AGRIASSIST-RX: 60-QUESTION ROUTING EVALUATION")
    print("=" * 65)

    print("\nThis tests routing, not answer quality.\n")

    for test in TESTS:

        actual = "error"
        status = "ERROR"
        details = ""

        try:

            output = choose_tool(
                make_state(test["question"])
            )

            actual = output.get(
                "selected_tool", "missing"
            )

            details = output.get(
                "answer", ""
            )

            if is_router_error(details):

                status = "ERROR"

            elif actual == test["expected"]:

                status = "PASS"

            else:

                status = "FAIL"

        except Exception as error:

            details = (
                f"{type(error).__name__}: {error}"
            )

        record = {
            "id": test["id"],
            "question": test["question"],
            "expected": test["expected"],
            "actual": actual,
            "status": status,
            "details": details,
        }

        results.append(record)

        print(
            f"{test['id']:4} | "
            f"Expected: {test['expected']:<12} | "
            f"Actual: {actual:<12} | "
            f"{status}",
            flush=True,
        )

    # ---------------------------------------
    # 7. Calculate overall results
    # ---------------------------------------

    counts = Counter(
        item["status"]
        for item in results
    )

    passed = counts["PASS"]
    failed = counts["FAIL"]
    errors = counts["ERROR"]

    total = len(results)

    completed = passed + failed

    accuracy = (
        round(100 * passed / completed, 2)
        if completed
        else None
    )

    # ---------------------------------------
    # 8. Calculate results per route
    # ---------------------------------------

    per_route = {}

    for route in TEST_GROUPS:

        group = [
            result
            for result in results
            if result["expected"] == route
        ]

        group_counts = Counter(
            item["status"]
            for item in group
        )

        completed_group = (
            group_counts["PASS"]
            + group_counts["FAIL"]
        )

        per_route[route] = {
            "total": len(group),
            "passed": group_counts["PASS"],
            "failed": group_counts["FAIL"],
            "errors": group_counts["ERROR"],
            "accuracy_percent": (
                round(
                    100 * group_counts["PASS"]
                    / completed_group,
                    2
                )
                if completed_group
                else None
            ),
        }

    # ---------------------------------------
    # 9. Save evaluation report
    # ---------------------------------------

    output_dir = ROOT / "eval" / "results"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    summary = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "evaluation_type": (
            "New-question routing evaluation"
        ),
        "total_cases": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "accuracy_percent": accuracy,
        "accuracy_denominator": completed,
        "note": (
            "Routing only. Tools were not executed. "
            "No answer-quality or factual-grounding "
            "evaluation was performed. "
            "Errors are excluded from accuracy."
        ),
        "per_route": per_route,
        "results": results,
    }

    json_path = (
        output_dir
        / "routing_unseen_60.json"
    )

    csv_path = (
        output_dir
        / "routing_unseen_60.csv"
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
            ensure_ascii=False
        )

    with open(
        csv_path,
        "w",
        encoding="utf-8",
        newline=""
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

    # ---------------------------------------
    # 10. Display summary
    # ---------------------------------------

    print("\n" + "=" * 65)
    print("EVALUATION SUMMARY")
    print("=" * 65)

    print(f"Total questions: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"API/runtime errors: {errors}")

    print(
        "Routing accuracy (completed cases):",
        (
            f"{accuracy:.2f}%"
            if accuracy is not None
            else "Unavailable"
        )
    )

    print("\nPER-ROUTE RESULTS")

    for route, stats in per_route.items():

        print(
            f"{route:<12} "
            f"PASS={stats['passed']:2} "
            f"FAIL={stats['failed']:2} "
            f"ERROR={stats['errors']:2}"
        )

    print("\nFAILED OR ERROR CASES")

    problems = [
        result
        for result in results
        if result["status"] != "PASS"
    ]

    if not problems:
        print("None")

    for problem in problems:

        print(
            f"\n{problem['id']}: "
            f"{problem['question']}"
        )

        print(
            "Expected:",
            problem["expected"]
        )

        print(
            "Actual:",
            problem["actual"]
        )

        print(
            "Status:",
            problem["status"]
        )

        if problem["details"]:

            print(
                "Details:",
                problem["details"]
            )

    print("\nReports saved to:")
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
