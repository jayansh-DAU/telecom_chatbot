"""
Builds a merged retriever across all three Chroma collections:
  - faq     : FAQ entries (no chunking — 1 row = 1 doc)
  - tickets : resolved support tickets (no chunking — 1 ticket = 1 doc)
  - guides  : PDF guide chunks (RecursiveCharacterTextSplitter applied at ingest)
"""
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

CHROMA_DIR = "chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _annotate_score(document: Document, score: float | None) -> Document:
    metadata = dict(document.metadata or {})
    if score is not None:
        metadata["similarity_score"] = score
    return Document(page_content=document.page_content, metadata=metadata)


def _search_with_score(vectorstore: Chroma, query: str, k: int) -> list[Document]:
    try:
        results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    except Exception:
        results = [(document, None) for document in vectorstore.similarity_search(query, k=k)]

    return [_annotate_score(document, score) for document, score in results]


def build_retriever(
    k_faq: int = 3,
    k_tickets: int = 3,
    k_guides: int = 3,
) -> RunnableLambda:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    faq_store = Chroma(
        collection_name="faq",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    tickets_store = Chroma(
        collection_name="tickets",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    guides_store = Chroma(
        collection_name="guides",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    def retrieve(query: str) -> list[Document]:
        return (
            _search_with_score(faq_store, query, k_faq)
            + _search_with_score(tickets_store, query, k_tickets)
            + _search_with_score(guides_store, query, k_guides)
        )

    return RunnableLambda(retrieve)
