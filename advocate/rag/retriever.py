"""
retriever.py
Wraps ChromaDB with a cosine-similarity query interface.
Provides:
  - retrieve(query, n_results) -> list of result dicts
  - verify_citation(citation_text, threshold) -> bool  (for Rule Validity check)
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PERSIST_PATH", "./advocate/data/chroma_db")
COLLECTION_NAME = "employment_law"
RULE_VALIDITY_THRESHOLD = float(os.getenv("RULE_VALIDITY_THRESHOLD", "0.75"))

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
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def retrieve(query: str, n_results: int = 5, side: str = "") -> list[dict]:
    """
    Semantic search over the ChromaDB index.

    Args:
        query:     Natural language query string.
        n_results: Number of top chunks to return.
        side:      Optional label ("employer" | "employee") — used only for logging.

    Returns:
        List of dicts: {text, case_name, citation, court, date_filed, url, score}
        score is cosine similarity (0–1); higher = more relevant.
    """
    model = _get_model()
    collection = _get_collection()

    query_embedding = model.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]   # cosine distance (0=identical, 2=opposite)

    for doc, meta, dist in zip(docs, metas, distances):
        # ChromaDB cosine distance: similarity = 1 - distance/2  (for normalized vectors)
        similarity = max(0.0, 1.0 - dist / 2.0)
        output.append({
            "text": doc,
            "case_name": meta.get("case_name", "Unknown"),
            "citation": meta.get("citation", ""),
            "court": meta.get("court", ""),
            "date_filed": meta.get("date_filed", ""),
            "url": meta.get("url", ""),
            "score": round(similarity, 4),
        })

    return output


def verify_citation(citation_text: str, threshold: float = RULE_VALIDITY_THRESHOLD) -> tuple[bool, float]:
    """
    Programmatic Rule Validity check.
    Searches the RAG index for the citation string.
    Returns (is_valid, best_score).
    If best cosine similarity < threshold → invalid (hallucinated citation).
    """
    results = retrieve(citation_text, n_results=3)
    if not results:
        return False, 0.0
    best_score = max(r["score"] for r in results)
    return best_score >= threshold, round(best_score, 4)


def collection_size() -> int:
    """Return number of chunks currently in the index."""
    try:
        return _get_collection().count()
    except Exception:
        return 0
