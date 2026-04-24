"""ChromaDB-backed retrieval helpers for ADVOCATE."""

from __future__ import annotations

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from advocate.settings import get_chroma_persist_path, get_setting

load_dotenv()

CHROMA_PATH = get_chroma_persist_path()
COLLECTION_NAME = "employment_law"
RULE_VALIDITY_THRESHOLD = float(get_setting("RULE_VALIDITY_THRESHOLD", "0.75"))

_model: SentenceTransformer | None = None
_collection = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        try:
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            _collection = client.get_collection(COLLECTION_NAME)
        except Exception:
            return None
    return _collection


def index_ready() -> bool:
    collection = _get_collection()
    return collection is not None and collection.count() > 0


def retrieve(query: str, n_results: int = 5, side: str = "") -> list[dict]:
    """Run semantic search over the local ChromaDB index."""
    del side
    collection = _get_collection()
    if collection is None:
        return []

    model = _get_model()
    query_embedding = model.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    documents = results["documents"][0]
    metadata_rows = results["metadatas"][0]
    distances = results["distances"][0]

    for document, metadata, distance in zip(documents, metadata_rows, distances):
        similarity = max(0.0, 1.0 - distance / 2.0)
        output.append(
            {
                "text": document,
                "case_name": metadata.get("case_name", "Unknown"),
                "citation": metadata.get("citation", ""),
                "court": metadata.get("court", ""),
                "date_filed": metadata.get("date_filed", ""),
                "url": metadata.get("url", ""),
                "score": round(similarity, 4),
            },
        )

    return output


def verify_citation(citation_text: str, threshold: float = RULE_VALIDITY_THRESHOLD) -> tuple[bool, float]:
    """Return whether the citation is grounded in the RAG index."""
    results = retrieve(citation_text, n_results=3)
    if not results:
        return False, 0.0
    best_score = max(result["score"] for result in results)
    return best_score >= threshold, round(best_score, 4)


def collection_size() -> int:
    collection = _get_collection()
    if collection is None:
        return 0
    try:
        return collection.count()
    except Exception:
        return 0
