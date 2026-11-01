# AgriAssist-RX Setup Guide

This guide explains how to install AgriAssist-RX and prepare its agricultural knowledge base.

The project was developed on Windows using VS Code and Python.

## 1. Install the Project

Clone the repository:

```powershell
git clone https://github.com/Muhammad-Usman-Shah/AgriAssist-RX.git
cd AgriAssist-RX
```

Create a Python environment:

```powershell
python -m venv .venv
```

Install the packages:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. Add the Gemini API Key

Create a `.env` file in the main project folder.

Add your Gemini API key:

```env
GEMINI_API_KEY=your_api_key_here
```

Keep this file private.

## 3. Prepare the Agricultural Documents

Create a folder named `raw_pdfs` inside `data`.

The folder path should be:

```text
data/raw_pdfs/
```

Place your agricultural PDF documents inside it.

The original project used four documents about wheat rust diseases. These files aren't included in the public repository.

You can use your own PDFs, but the answers will depend on the documents you provide.

The current ingestion script works with PDFs containing readable text. It doesn't perform OCR on scanned pages.

## 4. Extract Text from the PDFs

Run:

```powershell
.\.venv\Scripts\python.exe src\ingest.py
```

The script reads the PDF files, cleans the text, and keeps the document names and page numbers.

It saves the extracted text to:

```text
data/processed/extracted_pages.jsonl
```

## 5. Create Text Chunks

Run:

```powershell
.\.venv\Scripts\python.exe src\chunk.py
```

The script splits the extracted text into smaller sections.

Each chunk contains up to 400 words, with a 60-word overlap between consecutive chunks from the same page.

The output is saved to:

```text
data/processed/chunks.jsonl
```

## 6. Build the Knowledge Base

Run:

```powershell
.\.venv\Scripts\python.exe src\embed.py
```

The script uses the BGE-small embedding model to convert text chunks into numerical vectors.

It then stores the vectors and source information in ChromaDB.

The local database is saved in:

```text
data/chroma_db/
```

The embedding model may need to be downloaded during the first run, so an internet connection may be required.

Running this script again rebuilds the active knowledge collection from the current chunks.

## 7. Start the Assistant

Once the knowledge base is ready, run:

```powershell
.\.venv\Scripts\python.exe main.py
```

Try an agricultural question:

```text
What are the symptoms of wheat stem rust?
```

The RAG tool searches the documents and uses Gemini to prepare an answer.

Type `exit` to close the application.

## 8. Image Classification Setup

The image classifier uses a fine-tuned MobileNetV3-Small model.

The trained checkpoint and original image dataset aren't included in the repository.

To use image classification, you need the compatible model checkpoint and must update `MODEL_PATH` in:

```text
src/vision_tool.py
```

The checkpoint must match the architecture and class information expected by the code.

Without the checkpoint, the image classification tool cannot run.

The other tools can be prepared separately.

## 9. Things to Check

If the RAG tool cannot find information, first check that the PDFs contain readable text and that the ingestion, chunking, and embedding steps completed.

If an API request fails, check your internet connection, API key, and service availability.

If image classification fails, check the image path and model checkpoint location.

The project currently uses local files and paths. Some settings must be adjusted when running it on another computer.

## Current Setup Limits

The requirements file has been checked in the original development environment, but a full installation on a separate computer hasn't been tested.

The agricultural documents and model weights must also be obtained separately.

This guide covers the existing local setup. It doesn't provide a ready-to-run hosted version of AgriAssist-RX.