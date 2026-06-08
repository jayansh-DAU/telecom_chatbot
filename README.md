# RAG Telecom Chatbot

A Retrieval-Augmented Generation (RAG) customer care chatbot for telecom support. It answers questions about mobile connectivity, billing, SIM issues, and roaming by retrieving relevant context from three knowledge sources and generating responses with Qwen3-32B via Groq.

## Architecture

```
User question
     │
     ▼
Merged Retriever (top-k from each store)
  ├── ChromaDB · faq        (FAQ entries from CSV)
  ├── ChromaDB · tickets    (resolved support tickets from SQLite)
  └── ChromaDB · guides     (PDF guide chunks)
     │
     ▼
ChatPromptTemplate → Qwen3-32B (Groq) → Answer
```

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (runs locally via HuggingFace)  
**LLM:** `qwen/qwen3-32b` served by [Groq](https://groq.com)

## Project Structure

```
rag-telecom-chatbot/
├── app.py              # Streamlit web UI
├── main.py             # CLI entry point
├── rag_chain.py        # Builds the LangChain RAG chain
├── retriever.py        # Merges the three Chroma retrievers
├── ingest_faq.py       # Loads data/faq.csv → Chroma 'faq' collection
├── ingest_tickets.py   # Loads data/tickets.db → Chroma 'tickets' collection
├── ingest_pdf.py       # Loads data/telecom_guide.pdf → Chroma 'guides' collection
├── data/
│   ├── faq.csv             # FAQ question/answer pairs
│   ├── tickets.db          # SQLite database of resolved support tickets
│   ├── telecom_guide.pdf   # Telecom user guide (chunked at ingest)
│   ├── seed_tickets.py     # Script to seed the tickets database
│   └── generate_pdf.py     # Script to generate the telecom guide PDF
├── chroma_store/       # Persisted Chroma vector database (created at ingest)
├── pyproject.toml
├── uv.lock
└── .env.example
```

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- A [Groq API key](https://console.groq.com)
- A [HuggingFace token](https://huggingface.co/settings/tokens) (for downloading the embedding model)

## Setup

**1. Clone and install dependencies**

```bash
# RAG Telecom Chatbot

A Retrieval-Augmented Generation (RAG) customer-care chatbot for telecom support. It retrieves relevant context from FAQs, support tickets, and product guides, then composes concise, context-aware answers with a large language model.

## Features
- Ingests heterogeneous sources (CSV, SQLite, PDF) into a Chroma vector store
- Merged retriever that queries top-K from each collection for robust context
- Streamlit UI and simple CLI for exploration and testing

## Architecture

```
User question
     │
     ▼
Merged Retriever (top-k from each store)
  ├── ChromaDB · faq        (FAQ entries from CSV)
  ├── ChromaDB · tickets    (resolved support tickets from SQLite)
  └── ChromaDB · guides     (PDF guide chunks)
     │
     ▼
ChatPromptTemplate → LLM (Qwen3 / other) → Answer
```

**Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (local or HF)  
**LLM (example):** `qwen/qwen3-32b` (served via Groq or another provider)

## Repo layout

```
.
├── app.py              # Streamlit web UI
├── main.py             # CLI entry point
├── rag_chain.py        # Builds the RAG chain
├── retriever.py        # Merges the three Chroma retrievers
├── ingest_faq.py       # data/faq.csv -> Chroma 'faq'
├── ingest_tickets.py   # data/tickets.db -> Chroma 'tickets'
├── ingest_pdf.py       # data/telecom_guide.pdf -> Chroma 'guides'
├── data/               # small sample data + generators
├── chroma_store/       # persisted Chroma vector DB (DO NOT commit)
├── pyproject.toml
├── uv.lock
└── .env.example
```

## Prerequisites
- Python 3.11+
- `uv` (optional) or `pip` for installation
- Groq API key (if using Groq) and HuggingFace token (for model downloads)

## Quick start

1) Clone and install

```bash
git clone <repo-url>
cd 11_project_telecom_chatbot
uv sync        # or: pip install -e .
```

2) Configure env

```bash
copy .env.example .env    # Windows
# edit .env and add keys
```

3) Build the vector store (run ingests)

```bash
python ingest_faq.py
python ingest_tickets.py
python ingest_pdf.py
```

Only re-run an ingest when its source data changes. Ingests persist vectors under `chroma_store/`.

## Run

- Streamlit UI:

```bash
streamlit run app.py
```

- CLI:

```bash
python main.py
```

## Regenerate seed data

```bash
python data/seed_tickets.py
python data/generate_pdf.py
```

After regenerating sources, re-run the corresponding ingest script.

## What to include in the Git repo
- Source code: `app.py`, `main.py`, `rag_chain.py`, `retriever.py`, `ingest_*.py`
- Project metadata: `pyproject.toml`, `README.md`, `LICENSE`, `.env.example`
- Small sample data or truncated examples in `data/` (do not include full large datasets)

## What NOT to commit
- Secrets or environment files (`.env`, API keys)
- The full `chroma_store/` directory and `chroma.sqlite3` (large, derived data)
- Virtual environments, caches, compiled files (`venv/`, `__pycache__/`, `*.pyc`)

Add this `.gitignore` to the repo root (recommended):

```
# Python
__pycache__/
*.pyc
venv/
.venv/

# Editor
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Secrets
.env
*.env

# Chroma / vector DB
chroma_store/
chroma.sqlite3

# Logs / build
*.log
build/
dist/
```

## Notes
- Keep `chroma_store/` out of source control — document how to rebuild it using the ingest scripts.
- Use Git LFS or external storage for any large binary files you must share.

## Troubleshooting
- If the embeddings download fails, ensure `HF_TOKEN` in `.env` is valid.
- If you see empty responses, re-run ingests and verify `chroma_store/` populated.

---
If you'd like, I can add the `.gitignore` file and update `.env.example` with example keys. See [README.md](README.md) for the updated file.
