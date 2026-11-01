
"""
AgriAssist-RX
Phase 6 - Unseen RAG Evaluation

30 new agricultural questions.

Evaluates:
1. Evidence-supported answers
2. Questions with uncertain document coverage
3. Appropriate refusals
4. Citation availability
5. API failures

Factual correctness and citation faithfulness
must be verified manually.

Run in batches to avoid Gemini rate limits.
"""

import argparse
import json
import sys
import time

from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag import generate_answer


# --------------------------------------------------
# 30 unseen evaluation questions
# --------------------------------------------------

TESTS = [

    # Questions expected to be answerable from PDFs

    {
        "id": "U01",
        "category": "factual",
        "question": "Which wheat tissues usually develop leaf rust pustules?",
        "expected_behavior": "supported"
    },
    {
        "id": "U02",
        "category": "factual",
        "question": "What color are the lesions produced by wheat stripe rust?",
        "expected_behavior": "supported"
    },
    {
        "id": "U03",
        "category": "factual",
        "question": "How are stripe rust pustules arranged on wheat leaves?",
        "expected_behavior": "supported"
    },
    {
        "id": "U04",
        "category": "factual",
        "question": "Why is early detection important for controlling cereal rust diseases?",
        "expected_behavior": "supported"
    },
    {
        "id": "U05",
        "category": "factual",
        "question": "What happens when a rust fungus overcomes a wheat variety's resistance genes?",
        "expected_behavior": "supported"
    },
    {
        "id": "U06",
        "category": "factual",
        "question": "What is Ug99 and why is it important in wheat production?",
        "expected_behavior": "supported"
    },
    {
        "id": "U07",
        "category": "factual",
        "question": "Why can early stem rust infection cause serious grain losses?",
        "expected_behavior": "supported"
    },
    {
        "id": "U08",
        "category": "factual",
        "question": "How does lodging affect the harvesting of wheat infected with stem rust?",
        "expected_behavior": "supported"
    },
    {
        "id": "U09",
        "category": "comparison",
        "question": "Compare the lesion colors of stem rust and leaf rust.",
        "expected_behavior": "supported"
    },
    {
        "id": "U10",
        "category": "comparison",
        "question": "How can the arrangement of pustules help identify stripe rust?",
        "expected_behavior": "supported"
    },
    {
        "id": "U11",
        "category": "factual",
        "question": "Why might planting wheat varieties with different resistance genes reduce disease risk?",
        "expected_behavior": "supported"
    },
    {
        "id": "U12",
        "category": "factual",
        "question": "Why does the timing of fungicide application matter for stem rust control?",
        "expected_behavior": "supported"
    },
    {
        "id": "U13",
        "category": "factual",
        "question": "How do wheat rust surveillance systems support disease management?",
        "expected_behavior": "supported"
    },
    {
        "id": "U14",
        "category": "factual",
        "question": "What is the purpose of wheat rust trap nurseries?",
        "expected_behavior": "supported"
    },
    {
        "id": "U15",
        "category": "factual",
        "question": "Why are rust-resistant wheat seeds important for managing epidemics?",
        "expected_behavior": "supported"
    },
    {
        "id": "U16",
        "category": "factual",
        "question": "What role can field surveys play in a wheat rust early-warning system?",
        "expected_behavior": "supported"
    },
    {
        "id": "U17",
        "category": "factual",
        "question": "How can weather systems influence the movement of wheat rust fungi?",
        "expected_behavior": "supported"
    },
    {
        "id": "U18",
        "category": "factual",
        "question": "What plant characteristics help distinguish stem rust from stripe rust?",
        "expected_behavior": "supported"
    },

    # Coverage uncertain: assess using actual evidence

    {
        "id": "U19",
        "category": "coverage_boundary",
        "question": "What exact temperature is best for every type of wheat rust?",
        "expected_behavior": "review"
    },
    {
        "id": "U20",
        "category": "coverage_boundary",
        "question": "Which wheat variety guarantees complete protection against all rust strains?",
        "expected_behavior": "review"
    },
    {
        "id": "U21",
        "category": "coverage_boundary",
        "question": "Can stripe rust and leaf rust infect the same field at the same time?",
        "expected_behavior": "review"
    },
    {
        "id": "U22",
        "category": "coverage_boundary",
        "question": "How many days after infection do stripe rust symptoms first become visible?",
        "expected_behavior": "review"
    },
    {
        "id": "U23",
        "category": "coverage_boundary",
        "question": "Does wheat stripe rust always cause more yield loss than stem rust?",
        "expected_behavior": "review"
    },
    {
        "id": "U24",
        "category": "coverage_boundary",
        "question": "Can wheat rust be diagnosed with complete certainty from leaf color alone?",
        "expected_behavior": "review"
    },

    # Questions that require missing or live information

    {
        "id": "U25",
        "category": "refusal",
        "question": "What is the live wheat price in Multan right now?",
        "expected_behavior": "refuse"
    },
    {
        "id": "U26",
        "category": "refusal",
        "question": "How much rain will fall on my farm tomorrow?",
        "expected_behavior": "refuse"
    },
    {
        "id": "U27",
        "category": "refusal",
        "question": "What is the exact soil nitrogen level in my field today?",
        "expected_behavior": "refuse"
    },
    {
        "id": "U28",
        "category": "refusal",
        "question": "What was my fertilizer purchase cost last month?",
        "expected_behavior": "refuse"
    },
    {
        "id": "U29",
        "category": "refusal",
        "question": "What is the current inventory of wheat seeds in my storage room?",
        "expected_behavior": "refuse"
    },
    {
        "id": "U30",
        "category": "refusal",
        "question": "What is the exact disease status of my wheat field without any observations?",
        "expected_behavior": "refuse"
    }
]


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def error_detected(text):

    value = str(text).lower()

    terms = [
        "resource_exhausted",
        "quota exceeded",
        "rate limit",
        "429 ",
        "503 unavailable",
        "ai router is temporarily unavailable",
        "service is temporarily unavailable"
    ]

    return any(term in value for term in terms)


def utc_now():

    return datetime.now(
        timezone.utc
    ).isoformat()


def save_report(results, start, filename):

    output_dir = ROOT / "eval" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    report = {
        "project": "AgriAssist-RX",
        "evaluation": "30 unseen RAG questions",
        "timestamp_utc": utc_now(),
        "batch_start": start,
        "batch_size": len(results),
        "total_generated": sum(
            r["status"] == "GENERATED"
            for r in results
        ),
        "total_errors": sum(
            r["status"] == "ERROR"
            for r in results
        ),
        "note": (
            "This report collects responses only. "
            "Factual correctness, answer completeness, "
            "and citation faithfulness require "
            "manual source verification."
        ),
        "results": results
    }

    temporary_path = output_dir / (
        filename + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            default=str
        ),
        encoding="utf-8"
    )

    temporary_path.replace(output_path)

    return output_path


# --------------------------------------------------
# Main evaluation
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start",
        type=int,
        default=0
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=12
    )

    args = parser.parse_args()

    if args.start < 0:
        parser.error("--start must be >= 0")

    if args.limit < 1:
        parser.error("--limit must be >= 1")

    if args.interval < 0:
        parser.error("--interval must be >= 0")

    selected_tests = TESTS[
        args.start:args.start + args.limit
    ]

    if not selected_tests:
        parser.error("No questions in this range")

    print("=" * 60)
    print("AGRIASSIST-RX: UNSEEN RAG EVALUATION")
    print("=" * 60)

    results = []

    filename = (
        f"rag_unseen_start{args.start}"
        f"_count{len(selected_tests)}.json"
    )

    for index, test in enumerate(selected_tests):

        if index > 0:
            time.sleep(args.interval)

        print(
            f"\n[{index + 1}/{len(selected_tests)}] "
            f"{test['id']}"
        )

        print("Question:", test["question"])

        try:

            response = generate_answer(
                test["question"]
            )

            if not isinstance(response, dict):
                raise TypeError(
                    "Expected a dictionary from RAG"
                )

            answer = str(
                response.get("answer", "")
            )

            supported = response.get(
                "supported"
            )

            sources = response.get(
                "sources", []
            ) or []

            status = (
                "ERROR"
                if error_detected(answer)
                else "GENERATED"
            )

            error = (
                answer if status == "ERROR" else ""
            )

        except Exception as exc:

            status = "ERROR"
            error = (
                f"{type(exc).__name__}: {exc}"
            )

            answer = ""
            supported = None
            sources = []

        result = {
            "id": test["id"],
            "category": test["category"],
            "question": test["question"],
            "expected_behavior": test[
                "expected_behavior"
            ],
            "status": status,
            "supported": supported,
            "answer": answer,
            "sources": sources,
            "error": error,
            "behavior_correct": None,
            "answer_correct": None,
            "citation_verified": None,
            "faithful_to_sources": None,
            "completeness": None,
            "review_notes": ""
        }

        results.append(result)

        print("Status:", status)
        print("Supported:", supported)
        print("Answer:", answer)
        print(
            "Sources:",
            json.dumps(
                sources,
                ensure_ascii=False,
                default=str
            )
        )

        if error:
            print("Error:", error)

        # Save results after every question.
        # This preserves progress if the run stops.
        saved_path = save_report(
            results,
            args.start,
            filename
        )

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print("Questions processed:", len(results))

    print(
        "Generated:",
        sum(
            r["status"] == "GENERATED"
            for r in results
        )
    )

    print(
        "Errors:",
        sum(
            r["status"] == "ERROR"
            for r in results
        )
    )

    print("Manual quality review: Pending")
    print("Report saved:", saved_path)


if __name__ == "__main__":
    main()
