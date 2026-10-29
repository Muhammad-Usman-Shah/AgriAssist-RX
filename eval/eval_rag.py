
"""
AgriAssist-RX
Phase 3: Resumable RAG Evaluation

Features:
- Preserves previous successful results
- Retries only failed or missing questions
- Handles temporary Gemini API errors
- Saves progress after each question
- Reports behavior accuracy and request completion
"""

import json
import sys
import time
import shutil
from pathlib import Path


# ==========================================
# STEP 1: Project paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_FOLDER = PROJECT_ROOT / "eval"
SRC_FOLDER = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_FOLDER))

from rag import generate_answer, client


RESULTS_FILE = EVAL_FOLDER / "rag_evaluation_results.json"
BACKUP_FILE = EVAL_FOLDER / "rag_evaluation_results_backup.json"

MAX_ATTEMPTS = 3
INITIAL_WAIT = 5


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
# STEP 3: Load previous results
# ==========================================

def load_previous_results():

    if not RESULTS_FILE.exists():
        print("No previous results found.")
        return {}

    # Make one backup before updating the report.
    if not BACKUP_FILE.exists():
        shutil.copy2(RESULTS_FILE, BACKUP_FILE)
        print("Backup created:", BACKUP_FILE.name)

    with RESULTS_FILE.open(
        "r", encoding="utf-8"
    ) as file:
        data = json.load(file)

    previous = {}

    for record in data.get("results", []):
        if "id" in record:
            previous[record["id"]] = record

    return previous


# ==========================================
# STEP 4: Calculate and save metrics
# ==========================================

def save_results(records):

    completed = [
        r for r in records
        if r.get("status") == "completed"
    ]

    correct = sum(
        r.get("behavior_correct") is True
        for r in completed
    )

    total = len(TEST_QUESTIONS)
    completed_count = len(completed)

    summary = {
        "total_questions": total,
        "completed_requests": completed_count,
        "failed_requests": total - completed_count,
        "correct_behavior": correct,
        "behavior_accuracy": (
            correct / completed_count
            if completed_count else None
        ),
        "overall_behavior_success_rate": (
            correct / total
        ),
        "note": (
            "These metrics assess answer/refusal "
            "behavior, not factual answer accuracy."
        )
    }

    report = {
        "summary": summary,
        "results": records
    }

    # Write to a temporary file before replacing
    # the report, reducing the risk of partial saves.
    temp_file = RESULTS_FILE.with_suffix(".json.tmp")

    with temp_file.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    temp_file.replace(RESULTS_FILE)

    return summary


# ==========================================
# STEP 5: Identify temporary API errors
# ==========================================

def is_temporary_error(error):

    status = getattr(error, "code", None)

    if status is None:
        status = getattr(error, "status_code", None)

    message = str(error)

    return (
        str(status) in ("429", "500", "502", "503", "504")
        or any(
            marker in message
            for marker in (
                "429 RESOURCE_EXHAUSTED",
                "500 INTERNAL",
                "502 BAD_GATEWAY",
                "503 UNAVAILABLE",
                "504 DEADLINE_EXCEEDED"
            )
        )
    )


# ==========================================
# STEP 6: Retry a question when appropriate
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

            temporary = is_temporary_error(error)

            print(
                f"Attempt {attempt}/{MAX_ATTEMPTS} "
                f"failed: {error}"
            )

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

            wait_seconds = INITIAL_WAIT * (2 ** (attempt - 1))

            print(
                f"Temporary API error. "
                f"Retrying after {wait_seconds} seconds..."
            )

            time.sleep(wait_seconds)


# ==========================================
# STEP 7: Resume evaluation
# ==========================================

def main():

    print("=" * 60)
    print("AGRIASSIST-RX: RESUMABLE RAG EVALUATION")
    print("=" * 60)

    previous = load_previous_results()

    records_by_id = {}

    for test in TEST_QUESTIONS:

        question_id = test["id"]
        old_record = previous.get(question_id)

        if (
            old_record
            and old_record.get("status") == "completed"
            and old_record.get("question") == test["question"]
            and old_record.get("expected_supported")
                == test["expected_supported"]
        ):
            records_by_id[question_id] = old_record

    print(
        "\nPreviously completed:",
        len(records_by_id)
    )

    pending = [
        test for test in TEST_QUESTIONS
        if test["id"] not in records_by_id
    ]

    print("Questions remaining:", len(pending))

    def ordered_records():
        return [
            records_by_id[test["id"]]
            for test in TEST_QUESTIONS
            if test["id"] in records_by_id
        ]

    try:
        for index, test in enumerate(pending, start=1):

            print("\n" + "-" * 60)
            print(
                f"Evaluating {test['id']} "
                f"({index}/{len(pending)})"
            )
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

            save_results(ordered_records())

            if index < len(pending):
                time.sleep(5)

    finally:
        summary = save_results(ordered_records())
        client.close()

    print("\n" + "=" * 60)
    print("FINAL EVALUATION SUMMARY")
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

    print("\nResults saved to:", RESULTS_FILE)


if __name__ == "__main__":
    main()
