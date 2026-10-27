
"""
AgriAssist-RX: Hybrid Agricultural Retrieval

Combines:
1. Dense retrieval (BGE-small + ChromaDB)
2. BM25 keyword retrieval
3. Reciprocal Rank Fusion (RRF)

Returns ranked agricultural evidence with citations.
"""

import json
import re
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = (
    PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
)

DB_FOLDER = PROJECT_ROOT / "data" / "chroma_db"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "agricultural_knowledge"

RRF_K = 60


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase search tokens."""
    return re.findall(r"\b\w+\b", text.lower())


class HybridRetriever:
    """Retrieve evidence using dense search, BM25, and RRF."""

    def __init__(self):
        if not CHUNKS_FILE.exists():
            raise FileNotFoundError(
                "chunks.jsonl is missing. Run chunk.py first."
            )

        with CHUNKS_FILE.open(
            "r", encoding="utf-8"
        ) as file:
            self.chunks = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

        if not self.chunks:
            raise ValueError("No document chunks found.")

        self.chunk_lookup = {
            chunk["chunk_id"]: chunk
            for chunk in self.chunks
        }

        if len(self.chunk_lookup) != len(self.chunks):
            raise ValueError("Duplicate chunk IDs found.")

        tokenized_corpus = [
            tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

        print("Loading embedding model...")
        self.model = SentenceTransformer(
            MODEL_NAME,
            device="cpu"
        )

        client = chromadb.PersistentClient(
            path=str(DB_FOLDER)
        )

        self.collection = client.get_collection(
            name=COLLECTION_NAME
        )

        if self.collection.count() != len(self.chunks):
            raise ValueError(
                "ChromaDB and chunks.jsonl counts differ. "
                "Run embed.py to rebuild the index."
            )

        print(f"Loaded {len(self.chunks)} knowledge chunks.")

    def dense_search(
        self,
        question: str,
        top_k: int = 10
    ) -> list[str]:
        """Return chunk IDs ranked by semantic similarity."""

        embedding = self.model.encode(
            [question],
            normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=embedding,
            n_results=min(top_k, self.collection.count())
        )

        return results["ids"][0]

    def bm25_search(
        self,
        question: str,
        top_k: int = 10
    ) -> list[str]:
        """Return chunk IDs ranked by keyword relevance."""

        tokens = tokenize(question)

        scores = self.bm25.get_scores(tokens)

        ranked = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True
        )

        return [
            self.chunks[index]["chunk_id"]
            for index in ranked[:top_k]
            if scores[index] > 0
        ]

    def hybrid_search(
        self,
        question: str,
        top_k: int = 5
    ) -> list[dict]:
        """Combine dense and BM25 rankings using RRF."""

        candidate_k = min(
            max(top_k, 10),
            len(self.chunks)
        )

        dense_ids = self.dense_search(
            question, candidate_k
        )

        bm25_ids = self.bm25_search(
            question, candidate_k
        )

        scores = {}

        for ranking in (dense_ids, bm25_ids):
            for rank, chunk_id in enumerate(
                ranking, start=1
            ):
                scores[chunk_id] = (
                    scores.get(chunk_id, 0.0)
                    + 1.0 / (RRF_K + rank)
                )

        ranked_ids = sorted(
            scores,
            key=lambda chunk_id: scores[chunk_id],
            reverse=True
        )

        output = []

        for chunk_id in ranked_ids[:top_k]:
            chunk = self.chunk_lookup[chunk_id].copy()

            chunk["rrf_score"] = scores[chunk_id]

            output.append(chunk)

        return output


def main():
    """Start an interactive agricultural search."""

    print("=" * 55)
    print("AgriAssist-RX: Hybrid Knowledge Retrieval")
    print("=" * 55)

    retriever = HybridRetriever()

    while True:
        question = input(
            "\nAsk an agriculture question "
            "(or type exit): "
        ).strip()

        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not question:
            continue

        results = retriever.hybrid_search(
            question,
            top_k=3
        )

        print("\nTop hybrid retrieval results:\n")

        for index, item in enumerate(
            results, start=1
        ):
            print(f"--- Result {index} ---")
            print("Source:", item["source"])
            print("Page:", item["page"])
            print(
                "RRF score:",
                round(item["rrf_score"], 5)
            )
            print("Text:", item["text"][:700])
            print("-" * 55)


if __name__ == "__main__":
    main()
