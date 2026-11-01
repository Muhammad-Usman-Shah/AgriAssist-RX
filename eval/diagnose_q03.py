
"""
AgriAssist-RX
Phase 6 - Q03 Retrieval Diagnostic

Investigates why the RAG system refused:
"How does wheat stripe rust spread?"

Runs dense, BM25 and hybrid retrieval.
Does not use Gemini or modify the knowledge base.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retriever import HybridRetriever


QUESTION = "How does wheat stripe rust spread?"
TOP_K = 5


def show_results(title, chunk_ids, retriever):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)

    if not chunk_ids:
        print("No results found.")
        return

    for rank, chunk_id in enumerate(chunk_ids, start=1):
        chunk = retriever.chunk_lookup[chunk_id]

        print(f"\nRANK {rank}")
        print("Source:", chunk.get("source"))
        print("Page:", chunk.get("page"))
        print("Chunk ID:", chunk_id)
        print("Text:")
        print(chunk.get("text", ""))


def main():
    print("Loading AgriAssist-RX retriever...")

    retriever = HybridRetriever()

    print("\nQUESTION:", QUESTION)
    print("Top K:", TOP_K)

    dense_ids = retriever.dense_search(
        QUESTION,
        top_k=TOP_K
    )

    bm25_ids = retriever.bm25_search(
        QUESTION,
        top_k=TOP_K
    )

    hybrid_results = retriever.hybrid_search(
        QUESTION,
        top_k=TOP_K
    )

    hybrid_ids = [
        result["chunk_id"]
        for result in hybrid_results
    ]

    show_results(
        "DENSE RETRIEVAL",
        dense_ids,
        retriever
    )

    show_results(
        "BM25 RETRIEVAL",
        bm25_ids,
        retriever
    )

    show_results(
        "HYBRID RETRIEVAL",
        hybrid_ids,
        retriever
    )

    output_file = (
        ROOT / "eval" / "results" /
        "q03_retrieval_diagnostic.txt"
    )

    print("\nDiagnostic completed.")
    print("No project files were modified.")


if __name__ == "__main__":
    main()
