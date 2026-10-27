
"""
AgriAssist-RX: Semantic Agricultural Search

Searches agricultural knowledge stored in ChromaDB
and returns relevant passages with source citations.
"""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DB_FOLDER = PROJECT_ROOT / "data" / "chroma_db"

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def main():
    print("=" * 55)
    print("AgriAssist-RX: Agricultural Knowledge Search")
    print("=" * 55)

    # Load the same embedding model used during indexing.
    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    # Connect to our existing ChromaDB database.
    client = chromadb.PersistentClient(path=str(DB_FOLDER))

    try:
        collection = client.get_collection(
            name="agricultural_knowledge"
        )
    except Exception as error:
        print("Could not open agricultural knowledge collection.")
        print("Run embed.py first.")
        print(error)
        return

    print(f"Knowledge base: {collection.count()} chunks")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("Ask an agriculture question: ").strip()

        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not question:
            continue

        # Convert the question into an embedding.
        query_embedding = model.encode(
            [question],
            normalize_embeddings=True
        ).tolist()

        # Search for the three most similar passages.
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(3, collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        print("\nMost relevant passages:\n")

        for index, (text, metadata, distance) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ),
            start=1
        ):
            print(f"--- Result {index} ---")
            print(f"Source: {metadata['source']}")
            print(f"Page: {metadata['page']}")
            print(f"Cosine distance: {distance:.4f}")
            print(f"Text: {text[:700]}...")
            print()

        print("-" * 55)


if __name__ == "__main__":
    main()
