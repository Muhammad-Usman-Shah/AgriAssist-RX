
"""
AgriAssist-RX
Phase 6 - Step 5

Recheck API errors from the 60-question evaluation.

Purpose:
1. Read the previous routing evaluation.
2. Find cases marked ERROR.
3. Retest only those cases.
4. Space out Gemini API requests.
5. Save results without overwriting the baseline.

This checks ROUTING ONLY.
It does not evaluate answer accuracy.
"""

import sys
import json
import time

from pathlib import Path
from datetime import datetime, timezone


# ==========================================
# 1. Project paths
# ==========================================

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

from agent_csv import choose_tool

REPORT_DIR = ROOT / "eval" / "results"

INPUT_REPORT = (
    REPORT_DIR / "routing_unseen_60.json"
)

OUTPUT_REPORT = (
    REPORT_DIR / "routing_api_recheck.json"
)

# Space requests to reduce rate-limit errors.
REQUEST_INTERVAL_SECONDS = 10


# ==========================================
# 2. Prepare LangGraph state
# ==========================================

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


# ==========================================
# 3. Identify API errors
# ==========================================

def is_api_error(message):

    text = str(message).lower()

    indicators = [
        "router is temporarily unavailable",
        "429 resource_exhausted",
        "503 unavailable",
        "quota exceeded",
        "resource_exhausted",
        "api key is missing",
    ]

    return any(
        item in text
        for item in indicators
    )


# ==========================================
# 4. Run unfinished evaluations
# ==========================================

def main():

    if not INPUT_REPORT.is_file():

        print(
            "Previous report not found:",
            INPUT_REPORT
        )

        return

    with open(
        INPUT_REPORT,
        "r",
        encoding="utf-8"
    ) as file:

        previous = json.load(file)

    # Only cases previously marked ERROR.

    error_cases = [
        item
        for item in previous["results"]
        if item["status"] == "ERROR"
    ]

    print("=" * 60)
    print("AGRIASSIST-RX: API ERROR RECHECK")
    print("=" * 60)

    print(
        f"Previously unfinished cases: "
        f"{len(error_cases)}"
    )

    if not error_cases:

        print("No API errors to recheck.")
        return

    results = []

    for index, case in enumerate(error_cases):

        # Avoid rapid consecutive API requests.

        if index > 0:

            time.sleep(
                REQUEST_INTERVAL_SECONDS
            )

        question = case["question"]

        expected = case["expected"]

        print(
            f"\nTesting {case['id']}...",
            flush=True
        )

        try:

            output = choose_tool(
                make_state(question)
            )

            actual = output.get(
                "selected_tool",
                "missing"
            )

            details = output.get(
                "answer",
                ""
            )

            if is_api_error(details):

                status = "ERROR"

            elif actual == expected:

                status = "PASS"

            else:

                status = "FAIL"

        except Exception as error:

            actual = "error"

            status = "ERROR"

            details = (
                f"{type(error).__name__}: {error}"
            )

        result = {
            "id": case["id"],
            "question": question,
            "expected": expected,
            "actual": actual,
            "status": status,
            "details": details,
        }

        results.append(result)

        print(
            f"Expected: {expected}"
        )

        print(
            f"Actual: {actual}"
        )

        print(
            f"Status: {status}"
        )

        if status == "ERROR":

            print(
                "API/runtime issue detected."
            )

    # ======================================
    # 5. Summarize results
    # ======================================

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

    report = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "evaluation": "API error recheck",
        "previous_report": INPUT_REPORT.name,
        "total_rechecked": len(results),
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "request_interval_seconds": (
            REQUEST_INTERVAL_SECONDS
        ),
        "note": (
            "Retests previously unfinished "
            "routing cases. Original results "
            "are preserved. Answer correctness "
            "was not evaluated."
        ),
        "results": results,
    }

    with open(
        OUTPUT_REPORT,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 60)
    print("API RECHECK SUMMARY")
    print("=" * 60)

    print(
        f"Questions rechecked: {len(results)}"
    )

    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"API/runtime errors: {errors}")

    print("\nReport saved:")
    print(OUTPUT_REPORT)


if __name__ == "__main__":
    main()
