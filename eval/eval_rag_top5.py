
"""
AgriAssist-RX
Phase 3: Top-5 RAG Evaluation (Resumable)

Features:
- Evaluates 10 agricultural and unsupported questions
- Uses the existing RAG chatbot with Top-5 retrieval
- Preserves previously completed questions
- Retries only failed or missing questions
- Handles temporary Gemini API errors
- Saves results after every question
- Calculates behavior agreement and completion rate
"""

import json
import sys
import time
from pathlib import Path


# ==========================================
# STEP 1: Project configuration
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_FOLDER = PROJECT_ROOT / "eval"
SRC_FOLDER = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_FOLDER))

from rag import generate_answer, client, retrieve_evidence


OUTPUT_FILE = EVAL_FOLDER / "rag_evaluation_top5.json"

MAX_ATTEMPTS = 3
INITIAL_WAIT = 5
TOP_K = 5


# ==========================================
# STEP 2: Evaluation questions
# ==========================================

TEST_QUESTIONS = [
    {
        "id": "Q01",
        "question": "What is wheat rust disease?",
        "expected_supported": True
    },
    {
        "id": "Q02",
        "question": "How does wheat rust spread?",
        "expected_supported": True
    },
    {
        "id": "Q03",
        "question": "What are the symptoms of wheat stem rust?",
        "expected_supported": True
    },
    {
        "id": "Q04",
        "question": "Why are resistant wheat varieties important?",
        "expected_supported": True
    },
    {
        "id": "Q05",
        "question": "Can foliar fungicides help manage stem rust?",
        "expected_supported": True
    },
    {
        "id": "Q06",
        "question": "What is the capital of France?",
        "expected_supported": False
    },
    {
        "id": "Q07",
        "question": (
            "What exact pesticide dosage is legally "
            "recommended for wheat rust in Punjab, "
            "Pakistan in 2026?"
        ),
        "expected_supported": False
    },
    {
        "id": "Q08",
        "question": "What is the current temperature in Lahore?",
        "expected_supported": False
    },
    {
        "id": "Q09",
        "question": (
            "What is today's official wheat "
            "market price in Pakistan?"
        ),
        "expected_supported": False
    },
    {
        "id": "Q10",
        "question": (
            "Can you diagnose the disease in my "
            "cotton leaf photograph? No photograph "
            "has been provided."
        ),
        "expected_supported": False
    }
]


# ==========================================
# STEP 3: Load previous evaluation results
# ==========================================

def load_previous_results():

    if not OUTPUT_FILE.exists():
        print("No previous Top-5 results found.")
        print("Starting a new evaluation.")
        return {}

    print("Loading previous Top-5 results...")

    with OUTPUT_FILE.open(
        "r", encoding="utf-8"
    ) as file:
        report = json.load(file)

    previous_results = {}

    for record in report.get("results", []):

        question_id = record.get("id")

        if question_id:
            previous_results[question_id] = record

    return previous_results


# ==========================================
# STEP 4: Save results and calculate metrics
# ==========================================

def save_report(records):

    completed = [
        record for record in records
        if record.get("status") == "completed"
    ]

    correct = sum(
        record.get("behavior_correct") is True
        for record in completed
    )

    completed_count = len(completed)
    total_count = len(TEST_QUESTIONS)

    behavior_accuracy = (
        correct / completed_count
        if completed_count > 0 else None
    )

    overall_success = correct / total_count

    summary = {
        "retrieval_top_k": TOP_K,
        "total_questions": total_count,
        "completed_requests": completed_count,
        "failed_requests": total_count - completed_count,
        "correct_behavior": correct,
        "behavior_accuracy": behavior_accuracy,
        "overall_behavior_success_rate": overall_success,
        "note": (
            "Behavior accuracy measures agreement "
            "between expected and actual answer/refusal "
            "decisions. It does not measure factual "
            "correctness or citation accuracy."
        )
    }

    report = {
        "summary": summary,
        "results": records
    }

    temporary_file = OUTPUT_FILE.with_suffix(".json.tmp")

    with temporary_file.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    temporary_file.replace(OUTPUT_FILE)

    return summary


# ==========================================
# STEP 5: Identify temporary API errors
# ==========================================

def is_temporary_error(error):

    error_code = getattr(error, "code", None)

    if error_code is None:
        error_code = getattr(
            error, "status_code", None
        )

    if str(error_code) in (
        "429", "500", "502", "503", "504"
    ):
        return True

    message = str(error).upper()

    temporary_errors = [
        "429 RESOURCE_EXHAUSTED",
        "500 INTERNAL",
        "502 BAD_GATEWAY",
        "503 UNAVAILABLE",
        "504 DEADLINE_EXCEEDED"
    ]

    return any(
        item in message
        for item in temporary_errors
    )


# ==========================================
# STEP 6: Evaluate one question
# ==========================================

def evaluate_question(test):

    question = test["question"]
    expected = test["expected_supported"]

    for attempt in range(1, MAX_ATTEMPTS + 1):

        try:
            result = generate_answer(question)

            actual = result["supported"]

            return {
                "id": test["id"],
                "question": question,
                "expected_supported": expected,
                "actual_supported": actual,
                "behavior_correct": actual == expected,
                "answer": result["answer"],
                "sources": result["sources"],
                "status": "completed",
                "attempts": attempt
            }

        except Exception as error:

            print(
                f"Attempt {attempt}/{MAX_ATTEMPTS} "
                f"failed: {error}"
            )

            temporary = is_temporary_error(error)

            if not temporary or attempt == MAX_ATTEMPTS:

                return {
                    "id": test["id"],
                    "question": question,
                    "expected_supported": expected,
                    "status": "error",
                    "error": str(error),
                    "behavior_correct": False,
                    "attempts": attempt
                }

            wait_seconds = (
                INITIAL_WAIT * (2 ** (attempt - 1))
            )

            print(
                "Temporary API error. "
                f"Retrying after {wait_seconds} seconds..."
            )

            time.sleep(wait_seconds)


# ==========================================
# STEP 7: Run or resume Top-5 evaluation
# ==========================================

def main():

    print("=" * 60)
    print("AGRIASSIST-RX: TOP-5 RAG EVALUATION")
    print("=" * 60)

    print("Total questions:", len(TEST_QUESTIONS))
    print("Retrieval depth:", TOP_K)

    # Verify the retriever accepts five passages.
    try:
        _, context = retrieve_evidence(
            "What are the symptoms of wheat stem rust?",
            top_k=TOP_K
        )

        if not context.strip():
            print("Warning: No retrieval context found.")

    except Exception as error:
        print("Retriever verification failed:", error)
        client.close()
        return

    previous_results = load_previous_results()

    # Preserve records only when the question,
    # expected decision, and status still match.
    records_by_id = {}

    for test in TEST_QUESTIONS:

        question_id = test["id"]
        old_record = previous_results.get(question_id)

        if not old_record:
            continue

        if (
            old_record.get("status") == "completed"
            and old_record.get("question")
                == test["question"]
            and old_record.get("expected_supported")
                == test["expected_supported"]
        ):
            records_by_id[question_id] = old_record

    pending_questions = [
        test for test in TEST_QUESTIONS
        if test["id"] not in records_by_id
    ]

    print(
        "\nPreviously completed:",
        len(records_by_id)
    )

    print(
        "Questions remaining:",
        len(pending_questions)
    )

    def ordered_records():

        return [
            records_by_id[test["id"]]
            for test in TEST_QUESTIONS
            if test["id"] in records_by_id
        ]

    try:

        for index, test in enumerate(
            pending_questions, start=1
        ):

            print("\n" + "-" * 60)

            print(
                f"Question {index}/{len(pending_questions)}"
            )

            print("ID:", test["id"])
            print(test["question"])

            record = evaluate_question(test)

            records_by_id[test["id"]] = record

            print("Status:", record["status"])

            if record["status"] == "completed":

                print(
                    "Behavior:",
                    "PASS" if record["behavior_correct"]
                    else "FAIL"
                )

                print(
                    "Sources:",
                    len(record["sources"])
                )

            # Save after every completed attempt.
            save_report(ordered_records())

            if index < len(pending_questions):
                time.sleep(5)

    finally:

        summary = save_report(ordered_records())
        client.close()

    print("\n" + "=" * 60)
    print("FINAL TOP-5 EVALUATION SUMMARY")
    print("=" * 60)

    print(
        "Completed requests:",
        summary["completed_requests"]
    )

    print(
        "Failed requests:",
        summary["failed_requests"]
    )

    print(
        "Correct behavior:",
        summary["correct_behavior"]
    )

    if summary["behavior_accuracy"] is not None:

        print(
            "Behavior accuracy:",
            f"{summary['behavior_accuracy']:.2%}"
        )

    print(
        "Overall behavior success:",
        f"{summary['overall_behavior_success_rate']:.2%}"
    )

    print("\nResults saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
