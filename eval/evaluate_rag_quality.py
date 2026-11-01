
"""
AgriAssist-RX
Phase 6 - RAG Answer Quality Evaluation

Tests the existing generate_answer() function.

Records:
- Generated answer
- Support/refusal flag
- PDF citations
- API errors
- Manual-review fields

Important:
This script does NOT automatically judge factual
correctness or citation faithfulness.
Those require checking the source documents.
"""

import argparse
import csv
import json
import sys
import time

from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

from rag import generate_answer


# -------------------------------------------
# Evaluation questions
# -------------------------------------------

TESTS = [
    {
        "id": "Q01",
        "question": "What are the symptoms of wheat stem rust?",
        "expected_behavior": "supported",
    },
    {
        "id": "Q02",
        "question": "What causes wheat leaf rust?",
        "expected_behavior": "supported",
    },
    {
        "id": "Q03",
        "question": "How does wheat stripe rust spread?",
        "expected_behavior": "supported",
    },
    {
        "id": "Q04",
        "question": "Which parts of wheat plants can stem rust infect?",
        "expected_behavior": "supported",
    },
    {
        "id": "Q05",
        "question": "What symptoms distinguish stem rust from leaf rust?",
        "expected_behavior": "supported",
    },
    {
        "id": "Q06",
        "question": "How can wheat rust affect crop growth and yield?",
        "expected_behavior": "supported",
    },
    {
        "id": "Q07",
        "question": "What environmental conditions encourage wheat rust?",
        "expected_behavior": "supported",
    },
    {
        "id": "Q08",
        "question": "What is the current market price of wheat in Lahore?",
        "expected_behavior": "refuse",
    },
    {
        "id": "Q09",
        "question": "What is the latest cotton market price today?",
        "expected_behavior": "refuse",
    },
    {
        "id": "Q10",
        "question": "What was the exact wheat yield on my farm yesterday?",
        "expected_behavior": "refuse",
    },
]


# -------------------------------------------
# Helpers
# -------------------------------------------

def api_error_message(message):

    text = str(message).lower()

    return any(
        phrase in text
        for phrase in [
            "resource_exhausted",
            "quota exceeded",
            "503 unavailable",
            "429 ",
            "rate limit",
        ]
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Zero-based question index",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of questions to evaluate",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=12.0,
        help="Seconds between Gemini requests",
    )

    args = parser.parse_args()

    if args.start < 0 or args.limit < 1:
        parser.error(
            "--start must be >= 0 and --limit >= 1"
        )

    if args.interval < 0:
        parser.error(
            "--interval must not be negative"
        )

    selected = TESTS[
        args.start:args.start + args.limit
    ]

    if not selected:
        parser.error(
            "No questions in the requested range"
        )

    print("=" * 60)
    print("AGRIASSIST-RX: RAG QUALITY EVALUATION")
    print("=" * 60)

    results = []

    for index, test in enumerate(selected):

        if index > 0:
            time.sleep(args.interval)

        print(
            f"\nEvaluating {test['id']}: "
            f"{test['question']}",
            flush=True,
        )

        try:
            output = generate_answer(
                test["question"]
            )

            if not isinstance(output, dict):
                raise TypeError(
                    "RAG returned a non-dictionary result"
                )

            answer = str(
                output.get("answer", "")
            )

            supported = output.get(
                "supported", None
            )

            sources = output.get(
                "sources", []
            ) or []

            status = (
                "ERROR"
                if api_error_message(answer)
                else "GENERATED"
            )

            error = (
                answer if status == "ERROR" else ""
            )

        except Exception as exc:

            answer = ""
            supported = None
            sources = []
            status = "ERROR"

            error = (
                f"{type(exc).__name__}: {exc}"
            )

        record = {
            "id": test["id"],
            "question": test["question"],
            "expected_behavior": (
                test["expected_behavior"]
            ),
            "status": status,
            "supported": supported,
            "answer": answer,
            "sources": sources,
            "error": error,

            # These must be evaluated manually.
            "behavior_correct": None,
            "answer_correct": None,
            "citation_verified": None,
            "faithful_to_sources": None,
            "review_notes": "",
        }

        results.append(record)

        print("Status:", status)
        print("Supported:", supported)
        print("Answer:", answer)
        print(
            "Sources:",
            json.dumps(
                sources,
                ensure_ascii=False,
                default=str,
            )
        )

        if error:
            print("Error:", error)

    # -------------------------------------------
    # Save results
    # -------------------------------------------

    out_dir = ROOT / "eval" / "results"

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    stem = (
        f"rag_quality_start{args.start}"
        f"_count{len(selected)}_{timestamp}"
    )

    json_file = out_dir / f"{stem}.json"
    csv_file = out_dir / f"{stem}.csv"

    report = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "test_type": (
            "RAG output and citation collection"
        ),
        "total_cases": len(results),
        "generated": sum(
            r["status"] == "GENERATED"
            for r in results
        ),
        "errors": sum(
            r["status"] == "ERROR"
            for r in results
        ),
        "note": (
            "Generated outputs are NOT automatically "
            "graded as correct. Manual verification "
            "against source PDFs is required."
        ),
        "results": results,
    }

    with open(
        json_file,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    fields = [
        "id",
        "question",
        "expected_behavior",
        "status",
        "supported",
        "answer",
        "sources",
        "error",
        "behavior_correct",
        "answer_correct",
        "citation_verified",
        "faithful_to_sources",
        "review_notes",
    ]

    with open(
        csv_file,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for result in results:
            row = dict(result)

            row["sources"] = json.dumps(
                row["sources"],
                ensure_ascii=False,
                default=str,
            )

            writer.writerow(row)

    print("\n" + "=" * 60)
    print("RAG EVALUATION SUMMARY")
    print("=" * 60)

    print("Total:", len(results))
    print("Generated:", report["generated"])
    print("Errors:", report["errors"])
    print("Manual verification: NOT YET DONE")

    print("\nReports saved:")
    print(json_file)
    print(csv_file)


if __name__ == "__main__":
    main()
