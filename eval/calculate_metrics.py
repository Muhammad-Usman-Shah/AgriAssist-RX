
"""
AgriAssist-RX: Retrieval Performance Evaluation

Calculates:
1. Hit@3
2. Mean Reciprocal Rank (MRR)

A passage is relevant when its label is 2.
"""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_FOLDER = PROJECT_ROOT / "eval"

RESULTS_FILE = EVAL_FOLDER / "retrieval_comparison.json"
LABELS_FILE = EVAL_FOLDER / "relevance_labels.json"
OUTPUT_FILE = EVAL_FOLDER / "metrics_results.json"

METHODS = ["dense", "bm25", "hybrid"]
RELEVANT_LABEL = 2
TOP_K = 3


def load_json(path: Path):
    """Read a JSON file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def calculate_question_metrics(results, labels):
    """Calculate Hit@3 and reciprocal rank for one question."""

    hit = 0
    reciprocal_rank = 0.0

    for rank, passage in enumerate(results[:TOP_K], start=1):
        chunk_id = passage["chunk_id"]

        if chunk_id not in labels:
            raise ValueError(
                f"Missing relevance label for {chunk_id}"
            )

        if labels[chunk_id] == RELEVANT_LABEL:
            hit = 1
            reciprocal_rank = 1.0 / rank
            break

    return hit, reciprocal_rank


def main():
    """Evaluate all retrieval methods."""

    comparisons = load_json(RESULTS_FILE)
    label_data = load_json(LABELS_FILE)
    labels_by_question = label_data["questions"]

    report = {}

    print("=" * 55)
    print("AgriAssist-RX: Retrieval Evaluation")
    print("=" * 55)

    for method in METHODS:
        total_hits = 0
        total_rr = 0.0
        question_results = []

        for item in comparisons:
            question_id = item["id"]

            if question_id not in labels_by_question:
                raise ValueError(
                    f"Missing labels for {question_id}"
                )

            labels = labels_by_question[question_id]
            passages = item["results"][method]

            hit, rr = calculate_question_metrics(
                passages, labels
            )

            total_hits += hit
            total_rr += rr

            question_results.append({
                "question_id": question_id,
                "hit_at_3": hit,
                "reciprocal_rank": rr
            })

        count = len(comparisons)

        if count == 0:
            raise ValueError("No evaluation questions found.")

        hit_at_3 = total_hits / count
        mrr = total_rr / count

        report[method] = {
            "hit_at_3": hit_at_3,
            "mrr": mrr,
            "questions": question_results
        }

        print(f"\nMethod: {method.upper()}")
        print(f"Hit@3: {hit_at_3:.4f}")
        print(f"MRR: {mrr:.4f}")

    output = {
        "evaluation_status": label_data["review_status"],
        "number_of_questions": len(comparisons),
        "relevant_label": RELEVANT_LABEL,
        "methods": report
    }

    with OUTPUT_FILE.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(output, file, indent=2)

    print("\n" + "=" * 55)
    print("EVALUATION COMPLETED")
    print("=" * 55)
    print("Saved to:", OUTPUT_FILE)
    print("Relevance labels are provisional.")


if __name__ == "__main__":
    main()
