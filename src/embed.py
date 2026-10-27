
"""
AgriAssist-RX: Embedding and Vector Database

Converts agricultural text chunks into embeddings
and stores them in a persistent ChromaDB database.
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

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def main():
    if not CHUNKS_FILE.exists():
        print("Chunks file not found. Run chunk.py first.")
        return

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        chunks = [json.loads(line) for line in file if line.strip()]

    if not chunks:
        print("No text chunks found.")
        return

    print(f"Loaded {len(chunks)} chunks.")

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    texts = [chunk["text"] for chunk in chunks]

    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    print("Opening ChromaDB...")
    client = chromadb.PersistentClient(path=str(DB_FOLDER))

    collection = client.get_or_create_collection(
        name="agricultural_knowledge",
        metadata={"hnsw:space": "cosine"}
    )

    collection.upsert(
        ids=[chunk["chunk_id"] for chunk in chunks],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[
            {
                "source": chunk["source"],
                "page": chunk["page"]
            }
            for chunk in chunks
        ]
    )

    print("\nEmbedding storage completed!")
    print("Chunks processed:", len(chunks))
    print("Records in database:", collection.count())
    print("Database folder:", DB_FOLDER)


if __name__ == "__main__":
    main()
