"""ChromaDB-backed retrieval helpers with graceful no-index fallbacks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dotenv import load_dotenv

from advocate.settings import get_chroma_persist_path, get_setting

load_dotenv()

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

CHROMA_PATH = get_chroma_persist_path()
COLLECTION_NAME = "employment_law"
RULE_VALIDITY_THRESHOLD = float(get_setting("RULE_VALIDITY_THRESHOLD", "0.75"))

_model: "SentenceTransformer | None" = None
_collection = None
_model_error: str | None = None
_collection_error: str | None = None


def _get_model() -> "SentenceTransformer | None":
    global _model, _model_error
    if _model is not None:
        return _model
    if _model_error is not None:
        return None

    try:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("all-MiniLM-L6-v2")
        return _model
    except Exception as exc:
        _model_error = str(exc)
        return None


def _get_collection():
    global _collection, _collection_error
    if _collection is not None:
        return _collection
    if _collection_error is not None:
        return None

    try:
        import chromadb

        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_collection(COLLECTION_NAME)
        return _collection
    except Exception as exc:
        _collection_error = str(exc)
        return None


def retrieval_backend_status() -> tuple[bool, str]:
    collection = _get_collection()
    if collection is not None:
        return True, "ChromaDB retrieval available"
    return False, _collection_error or "ChromaDB retrieval unavailable"


def index_ready() -> bool:
    collection = _get_collection()
    if collection is None:
        return False
    try:
        return collection.count() > 0
    except Exception:
        return False


def retrieve(query: str, n_results: int = 5, side: str = "") -> list[dict]:
    """Run semantic search over the local ChromaDB index."""
    del side
    collection = _get_collection()
    model = _get_model()
    if collection is None or model is None:
        return []

    try:
        query_embedding = model.encode([query])[0].tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return []

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
