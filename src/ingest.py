
"""
AgriAssist-RX: Agricultural Document Ingestion

This module:
1. Reads agricultural PDF documents.
2. Extracts text from each page.
3. Removes unwanted control characters.
4. Preserves source filenames and page numbers.
5. Saves cleaned text in JSONL format.
"""

import json
import re
import unicodedata
from pathlib import Path

import fitz  # PyMuPDF


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_FOLDER = PROJECT_ROOT / "data" / "raw_pdfs"

OUTPUT_FOLDER = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = OUTPUT_FOLDER / "extracted_pages.jsonl"


# --------------------------------------------------
# Text cleaning
# --------------------------------------------------

def clean_text(text: str) -> str:
    """Clean PDF text while preserving readable content."""

    # Normalize Unicode characters.
    text = unicodedata.normalize("NFKC", text)

    # Remove control characters, preserving newlines.
    text = "".join(
        char for char in text
        if char.isprintable() or char in "\n\t"
    )

    # Normalize whitespace.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove leading and trailing whitespace.
    return text.strip()


# --------------------------------------------------
# PDF text extraction
# --------------------------------------------------

def extract_pdf(pdf_path: Path) -> list[dict]:
    """Extract cleaned text and metadata from PDF pages."""

    extracted_pages = []

    with fitz.open(pdf_path) as document:

        for page_number, page in enumerate(
            document, start=1
        ):

            raw_text = page.get_text("text")

            cleaned_text = clean_text(raw_text)

            # Skip pages without readable text.
            if not cleaned_text:
                continue

            extracted_pages.append(
                {
                    "source": pdf_path.name,
                    "page": page_number,
                    "text": cleaned_text,
                }
            )

    return extracted_pages


# --------------------------------------------------
# Main ingestion pipeline
# --------------------------------------------------

def main() -> None:
    """Process agricultural PDFs and save extracted text."""

    print("=" * 55)
    print("AgriAssist-RX: Agricultural PDF Ingestion")
    print("=" * 55)

    # Ensure the required folders exist.
    PDF_FOLDER.mkdir(parents=True, exist_ok=True)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Find PDF files.
    pdf_files = sorted(
        file for file in PDF_FOLDER.iterdir()
        if file.is_file() and file.suffix.lower() == ".pdf"
    )

    if not pdf_files:
        print("\nNo PDF documents found.")
        print("Please add agricultural PDFs to:")
        print(PDF_FOLDER)
        return

    all_pages = []
    failed_files = []

    # Process documents.
    for pdf_path in pdf_files:

        try:
            pages = extract_pdf(pdf_path)

            all_pages.extend(pages)

            print(f"\nDocument: {pdf_path.name}")
            print(f"Readable pages: {len(pages)}")

        except Exception as error:

            failed_files.append(pdf_path.name)

            print(f"\nError processing {pdf_path.name}:")
            print(error)

    if not all_pages:
        print("\nNo readable text was extracted.")
        print("Check whether your PDFs contain selectable text.")
        return

    # Save text and metadata.
    with OUTPUT_FILE.open(
        "w", encoding="utf-8"
    ) as output:

        for page_data in all_pages:

            output.write(
                json.dumps(
                    page_data,
                    ensure_ascii=False
                ) + "\n"
            )

    print("\n" + "=" * 55)
    print("PDF INGESTION COMPLETED")
    print("=" * 55)

    print(f"Documents found: {len(pdf_files)}")
    print(f"Readable pages extracted: {len(all_pages)}")
    print(f"Documents with errors: {len(failed_files)}")
    print(f"Output file: {OUTPUT_FILE}")

    if failed_files:
        print("\nFailed documents:")
        for filename in failed_files:
            print(f"- {filename}")


if __name__ == "__main__":
    main()
