
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"

BATCHES = [
    "rag_unseen_start0_count5.json",
    "rag_unseen_start5_count5.json",
    "rag_unseen_start10_count5.json",
    "rag_unseen_start15_count5.json",
    "rag_unseen_start20_count5.json",
    "rag_unseen_start25_count5.json",
]

RETRY_FILE = "rag_unseen_start1_count1.json"


def load(filename):
    with (RESULTS / filename).open(
        "r", encoding="utf-8"
    ) as f:
        return json.load(f)


def main():
    original = []

    for filename in BATCHES:
        report = load(filename)
        original.extend(report["results"])

    ids = [item["id"] for item in original]

    expected = [f"U{i:02d}" for i in range(1, 31)]

    if ids != expected:
        raise ValueError(
            "Missing, duplicate or out-of-order questions."
        )

    retry_results = load(RETRY_FILE)["results"]

    if (
        len(retry_results) != 1
        or retry_results[0]["id"] != "U02"
    ):
        raise ValueError("Unexpected retry results.")

    initial_errors = sum(
        item["status"] == "ERROR"
        for item in original
    )

    retry = retry_results[0]

    if retry["status"] != "GENERATED":
        raise ValueError("U02 retry was not successful.")

    reviewed = []

    for item in original:
        record = dict(item)

        if record["id"] == "U02":
            record["initial_attempt"] = {
                "status": item["status"],
                "error": item["error"],
            }

            record["status"] = retry["status"]
            record["supported"] = retry["supported"]
            record["answer"] = retry["answer"]
            record["sources"] = retry["sources"]
            record["error"] = retry["error"]

        if record["id"] == "U15":
            judgment = "partial"
            notes = (
                "Overstates the ability of resistant "
                "seeds to prevent disease outbreaks."
            )

        elif record["id"] == "U21":
            judgment = "grounding_failure"
            notes = (
                "The cited fungicide passage does "
                "not directly establish simultaneous "
                "infection in one field."
            )

        else:
            judgment = "pass"
            notes = (
                "Passed the prior manual "
                "source-evidence review."
            )

        record["manual_judgment"] = judgment
        record["review_notes"] = notes

        reviewed.append(record)

    counts = {
        label: sum(
            r["manual_judgment"] == label
            for r in reviewed
        )
        for label in [
            "pass", "partial", "grounding_failure"
        ]
    }

    output = {
        "project": "AgriAssist-RX",
        "evaluation": "30-question unseen RAG pilot",
        "knowledge_chunks": 38,
        "unique_questions": 30,
        "first_attempt_generated": 30 - initial_errors,
        "first_attempt_errors": initial_errors,
        "completed_after_retry": sum(
            r["status"] == "GENERATED"
            for r in reviewed
        ),
        "manual_review_counts": counts,
        "note": (
            "Manual judgments were assigned during "
            "the question-by-question review. "
            "Results are limited to this curated "
            "test set and four source PDFs."
        ),
        "results": reviewed,
    }

    destination = (
        RESULTS / "rag_unseen_final_review.json"
    )

    destination.write_text(
        json.dumps(
            output, indent=2, ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("UNSEEN RAG EVALUATION CONSOLIDATED")
    print("Questions:", len(reviewed))
    print("First-attempt errors:", initial_errors)
    print(
        "Completed after retry:",
        output["completed_after_retry"]
    )
    print("Manual review:", counts)
    print("Saved:", destination)


if __name__ == "__main__":
    main()
