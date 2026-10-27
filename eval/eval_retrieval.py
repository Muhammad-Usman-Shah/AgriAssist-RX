
"""
AgriAssist-RX: Automated Retrieval Comparison

Compares dense, BM25 and hybrid retrieval results.
This is a diagnostic report, not an accuracy score.
"""

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retriever import HybridRetriever


QUESTIONS_FILE = (
    PROJECT_ROOT / "eval" / "retrieval_test_questions.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "eval" / "retrieval_comparison.json"
)


def main() -> None:
    """Compare retrieval methods for the test questions."""

    with QUESTIONS_FILE.open(
        "r", encoding="utf-8"
    ) as file:
        questions = json.load(file)

    retriever = HybridRetriever()
    report = []

    for item in questions:
        question = item["question"]

        print(f"\nQuestion: {question}")

        dense_ids = retriever.dense_search(
            question, top_k=3
        )

        bm25_ids = retriever.bm25_search(
            question, top_k=3
        )

        hybrid_results = retriever.hybrid_search(
            question, top_k=3
        )

        hybrid_ids = [
            result["chunk_id"]
            for result in hybrid_results
        ]

        methods = {
            "dense": dense_ids,
            "bm25": bm25_ids,
            "hybrid": hybrid_ids
        }

        comparison = {
            "id": item["id"],
            "question": question,
            "topic": item["topic"],
            "results": {}
        }

        for method, ids in methods.items():
            print(f"\n{method.upper()}:")

            passages = []

            for rank, chunk_id in enumerate(ids, start=1):
                chunk = retriever.chunk_lookup[chunk_id]

                passage = {
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "text": chunk["text"]
                }

                passages.append(passage)

                print(
                    f"  {rank}. {chunk['source']} "
                    f"(page {chunk['page']})"
                )

            comparison["results"][method] = passages

        report.append(comparison)

    with OUTPUT_FILE.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("\n" + "=" * 55)
    print("RETRIEVAL COMPARISON COMPLETED")
    print("=" * 55)
    print("Questions evaluated:", len(report))
    print("Methods compared: Dense, BM25, Hybrid")
    print("Results saved to:", OUTPUT_FILE)
    print("\nNote: Relevance labels are required")
    print("before calculating Hit@k or MRR.")


if __name__ == "__main__":
    main()
