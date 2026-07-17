import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag_chain import stream_answer_with_documents
from retriever import build_retriever

load_dotenv()

SAMPLE_QUESTIONS = [
    "Why is my mobile internet so slow?",
    "My calls keep dropping — what should I do?",
    "How do I activate international roaming?",
    "Why is my bill higher than usual this month?",
    "My phone shows SIM not detected after a restart",
    "How do I enable Wi-Fi calling?",
    "I was charged for roaming but had a bundle active",
    "How do I unlock my phone for another network?",
]

st.set_page_config(
    page_title="Telecom Support Chat",
    page_icon="📡",
    layout="centered",
)

@st.cache_resource
def get_retriever():
    return build_retriever()


def _document_name(metadata: dict) -> str:
    source = str(metadata.get("source", "")).lower()
    if source == "faq":
        faq_id = metadata.get("faq_id")
        return f"FAQ {faq_id}" if faq_id else "FAQ"
    if source == "ticket":
        ticket_id = metadata.get("ticket_id")
        return f"Resolved Ticket {ticket_id}" if ticket_id else "Resolved Ticket"
    if source == "guide":
        return "Telecom Guide"
    source_value = str(metadata.get("source", "Unknown"))
    return Path(source_value).name if source_value else "Unknown"


def _page_number(metadata: dict) -> str:
    page = metadata.get("page", metadata.get("page_number"))
    return str(page) if page is not None else "Not Available"


def _similarity_score(metadata: dict) -> str:
    score = metadata.get("similarity_score", metadata.get("relevance_score"))
    if isinstance(score, (int, float)):
        return f"{score:.2f}"
    return "Not Available"


def _render_retrieved_context(documents) -> None:
    if not documents:
        st.info("No retrieved context available.")
        return

    for index, document in enumerate(documents, start=1):
        metadata = document.metadata or {}

        st.markdown(f"**Chunk {index}**")
        st.markdown(f"**Document:** {_document_name(metadata)}")
        st.markdown(f"**Page:** {_page_number(metadata)}")
        st.markdown(f"**Similarity Score:** {_similarity_score(metadata)}")
        st.markdown("**Chunk Text:**")
        st.write(document.page_content)

        if index < len(documents):
            st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📡 Telecom Support")
    st.caption("Powered by RAG · Qwen3-32B on Groq")
    st.divider()

    st.markdown("**Sample questions**")
    st.caption("Click one to send it instantly.")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []

# ── Main ─────────────────────────────────────────────────────────────────────
st.title("Customer Care Assistant")
st.caption("Ask me anything about your mobile service — connectivity, billing, SIM, roaming, and more.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Resolve question from chat input or sidebar button click
question = st.chat_input("Describe your issue…")
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    retrieved_documents = get_retriever().invoke(question)

    with st.chat_message("assistant"):
        response = st.write_stream(
            stream_answer_with_documents(question, retrieved_documents)
        )

        with st.expander("🔍 Retrieved Context"):
            _render_retrieved_context(retrieved_documents)

    st.session_state.messages.append({"role": "assistant", "content": response})
