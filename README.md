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
│   ├── faq.csv         # FAQ question/answer pairs
│   ├── tickets.db      # SQLite database of resolved support tickets
│   └── telecom_guide.pdf # Telecom user guide (chunked at ingest)
├── chroma_store/       # Persisted Chroma vector database (created at ingest)
├── .env.example
├── .gitignore
├── pyproject.toml
└── uv.lock
```

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- A [Groq API key](https://console.groq.com)
- A [HuggingFace token](https://huggingface.co/settings/tokens) for downloading the embedding model

## Setup

1. Clone and install dependencies.

```bash
git clone <repo-url>
cd 11_project_telecom_chatbot
uv sync
```

If you prefer pip, install the project in editable mode instead.

2. Configure environment variables.

```bash
copy .env.example .env    # Windows
# edit .env and add your keys
```

3. Build the vector store.

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

## Regenerate Seed Data

```bash
python data/seed_tickets.py
python data/generate_pdf.py
```

After regenerating sources, re-run the corresponding ingest script.


- If the embeddings download fails, ensure `HF_TOKEN` in `.env` is valid.
- If you see empty responses, re-run the ingest scripts and verify `chroma_store/` was populated.
