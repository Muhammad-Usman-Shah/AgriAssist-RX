
"""
AgriAssist-RX: Agricultural Knowledge Base Indexing

Reads processed agricultural text chunks,
generates embeddings, and rebuilds the
ChromaDB collection with current records.
"""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = (
    PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
)

DB_FOLDER = PROJECT_ROOT / "data" / "chroma_db"

COLLECTION_NAME = "agricultural_knowledge"

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_chunks() -> list[dict]:
    """Load and validate text chunks from JSONL."""

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            "chunks.jsonl not found. Run chunk.py first."
        )

    chunks = []

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    if not chunks:
        raise ValueError("No chunks were found.")

    ids = [chunk["chunk_id"] for chunk in chunks]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs detected.")

    return chunks


def main() -> None:
    """Build the agricultural vector database."""

    print("=" * 55)
    print("AgriAssist-RX: Knowledge Base Indexing")
    print("=" * 55)

    chunks = load_chunks()

    print(f"\nLoaded {len(chunks)} text chunks.")

    # Load embedding model.
    print("\nLoading BGE-small embedding model...")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )

    texts = [chunk["text"] for chunk in chunks]

    # Create normalized embeddings.
    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    if len(embeddings) != len(chunks):
        raise RuntimeError("Embedding count mismatch.")

    print("\nConnecting to ChromaDB...")

    client = chromadb.PersistentClient(
        path=str(DB_FOLDER)
    )

    # Build a replacement collection first.
    # This preserves the existing collection if indexing fails.
    temporary_name = "agricultural_knowledge_build"

    existing = [
        collection.name
        for collection in client.list_collections()
    ]

    if temporary_name in existing:
        client.delete_collection(temporary_name)

    collection = client.create_collection(
        name=temporary_name,
        metadata={"hnsw:space": "cosine"}
    )

    collection.add(
        ids=[chunk["chunk_id"] for chunk in chunks],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[
            {
                "source": chunk["source"],
                "page": int(chunk["page"])
            }
            for chunk in chunks
        ]
    )

    if collection.count() != len(chunks):
        raise RuntimeError(
            "Database verification failed."
        )

    # Replace the old collection only after successful indexing.
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    collection.modify(name=COLLECTION_NAME)

    print("\n" + "=" * 55)
    print("KNOWLEDGE BASE REBUILT SUCCESSFULLY")
    print("=" * 55)
    print("Chunks indexed:", len(chunks))
    print("Database records:", collection.count())
    print("Database location:", DB_FOLDER)


if __name__ == "__main__":
    main()
