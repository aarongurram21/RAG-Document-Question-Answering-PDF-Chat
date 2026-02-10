# PDF RAG Q&A

A minimal Retrieval-Augmented Generation workflow for PDF question answering with citations using FAISS, Sentence Transformers, and Gradio.

## Features
- Upload a PDF and extract text
- Configurable text chunking (size and overlap)
- Sentence Transformer embeddings stored in FAISS
- Top-k similarity retrieval
- FLAN-T5 generation with source citations
- Gradio UI plus evaluation example prompts

## Project layout
- src/ingest.py — PDF parsing and chunking utilities
- src/retriever.py — FAISS-backed retriever
- src/qa_chain.py — QA chain with generation and prompt
- app.py — Gradio interface and orchestration

## Setup
```bash
pip install -r requirements.txt
python app.py
```

## Usage
1. Start the app: `python app.py`
2. Upload a PDF, set chunk size/overlap and top-k.
3. Ask a question; answers include the chunk IDs as citations.

## Notes
- Default embedding model: sentence-transformers/all-MiniLM-L6-v2
- Default generator: google/flan-t5-small (downloads on first run)
- If you swap models, ensure embedding dimensionality matches FAISS index.
