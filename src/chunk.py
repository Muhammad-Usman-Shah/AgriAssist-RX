
"""
AgriAssist-RX: Text Chunking

Reads extracted agricultural PDF pages and splits
them into overlapping text chunks.

Preserves source filename and page number.
"""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "extracted_pages.jsonl"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "chunks.jsonl"
)

CHUNK_SIZE = 400
OVERLAP = 60


def split_text(text: str) -> list[str]:
    """Split text into overlapping word-based chunks."""

    words = text.split()
    chunks = []

    step = CHUNK_SIZE - OVERLAP

    for start in range(0, len(words), step):
        chunk_words = words[start:start + CHUNK_SIZE]

        if not chunk_words:
            break

        chunks.append(" ".join(chunk_words))

        if start + CHUNK_SIZE >= len(words):
            break

    return chunks


def main() -> None:
    """Create chunks from extracted PDF pages."""

    if not INPUT_FILE.exists():
        print("Input file not found!")
        print("Run ingest.py first.")
        return

    total_chunks = 0

    with INPUT_FILE.open(
        "r", encoding="utf-8"
    ) as input_file, OUTPUT_FILE.open(
        "w", encoding="utf-8"
    ) as output_file:

        for line in input_file:
            page_data = json.loads(line)

            text_chunks = split_text(page_data["text"])

            for chunk_index, text in enumerate(
                text_chunks, start=1
            ):
                chunk = {
                    "chunk_id": (
                        f"{page_data['source']}"
                        f"_p{page_data['page']}"
                        f"_c{chunk_index}"
                    ),
                    "source": page_data["source"],
                    "page": page_data["page"],
                    "text": text,
                }

                output_file.write(
                    json.dumps(
                        chunk, ensure_ascii=False
                    ) + "\n"
                )

                total_chunks += 1

    print("Text chunking completed!")
    print("Total chunks created:", total_chunks)
    print("Output saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
