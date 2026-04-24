"""Build a local ChromaDB employment-law index from CourtListener opinions."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import chromadb
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from advocate.settings import get_chroma_persist_path, get_setting

load_dotenv()

BASE_URL = get_setting("COURTLISTENER_BASE_URL", "https://www.courtlistener.com/api/rest/v4")
CHROMA_PATH = get_chroma_persist_path()
RAW_DIR = Path("./advocate/data/raw_cases")
PROCESSED_DIR = Path("./advocate/data/processed_cases")
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
COLLECTION_NAME = "employment_law"
TARGET_OPINIONS = 70


def fetch_opinions(max_results: int = TARGET_OPINIONS) -> list[dict]:
    """Pull wrongful termination opinions from CourtListener."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = RAW_DIR / "opinions_cache.json"

    if cache_file.exists():
        print(f"[build_index] Loading {max_results} opinions from cache.")
        with cache_file.open(encoding="utf-8") as handle:
            opinions = json.load(handle)
        return opinions[:max_results]

    print("[build_index] Fetching opinions from CourtListener...")
    opinions: list[dict] = []
    url = f"{BASE_URL}/search/"
    params = {
        "q": "wrongful termination employment",
        "type": "o",
        "order_by": "score desc",
    }
    token = get_setting("COURTLISTENER_API_TOKEN", "")
    headers = {"User-Agent": "ADVOCATE-Research-Tool/1.0"}
    if token:
        headers["Authorization"] = f"Token {token}"
    else:
        print(
            "\n[build_index] WARNING: COURTLISTENER_API_TOKEN not set.\n"
            "  1. Register at https://www.courtlistener.com/sign-in/\n"
            "  2. Copy your API token from your profile\n"
            "  3. Add COURTLISTENER_API_TOKEN to your environment or Streamlit secrets\n",
        )

    search_results: list[dict] = []
    while url and len(search_results) < max_results:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        if response.status_code == 401:
            raise RuntimeError(
                "CourtListener returned 401 Unauthorized. Add COURTLISTENER_API_TOKEN before building the index.",
            )
        response.raise_for_status()
        payload = response.json()
        search_results.extend(payload.get("results", []))
        print(f"  found {len(search_results)} cases so far...")
        url = payload.get("next")
        params = {}
        time.sleep(1)

    search_results = search_results[:max_results]
    print(f"[build_index] Fetching full text for {len(search_results)} cases...")

    for search_result in search_results:
        opinions_list = search_result.get("opinions", [])
        if not opinions_list:
            continue

        opinion_id = opinions_list[0].get("id")
        opinion = {
            "case_name": search_result.get("caseName", ""),
            "citation": (
                search_result.get("citation", [""])[0]
                if isinstance(search_result.get("citation"), list) and search_result.get("citation")
                else search_result.get("citation", "")
            ),
            "court": search_result.get("court", ""),
            "date_filed": search_result.get("dateFiled", ""),
            "absolute_url": search_result.get("absolute_url", ""),
        }

        try:
            opinion_response = requests.get(f"{BASE_URL}/opinions/{opinion_id}/", headers=headers, timeout=10)
            if opinion_response.status_code == 200:
                opinion_payload = opinion_response.json()
                opinion["plain_text"] = opinion_payload.get("plain_text", "")
                opinion["html_with_citations"] = opinion_payload.get("html_with_citations", "")
                opinion["html"] = opinion_payload.get("html", "")
                opinions.append(opinion)
            time.sleep(0.5)
        except Exception as exc:
            print(f"  [warn] Failed to fetch opinion {opinion_id}: {exc}")

    with cache_file.open("w", encoding="utf-8") as handle:
        json.dump(opinions, handle, indent=2)
    print(f"[build_index] Saved {len(opinions)} raw opinions to cache.")
    return opinions


def extract_text(opinion: dict) -> str:
    for field in ("plain_text", "html_with_citations", "html", "xml_harvard"):
        text = opinion.get(field, "")
        if text and len(text) > 100:
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text
    return ""


def char_chunk(text: str, chunk_chars: int = CHUNK_SIZE * 4, overlap_chars: int = CHUNK_OVERLAP * 4) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_chars - overlap_chars
    return chunks


def build_chromadb_index(opinions: list[dict]) -> None:
    """Embed and store opinion chunks in ChromaDB."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)

    print("[build_index] Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    all_chunks: list[str] = []
    all_ids: list[str] = []
    all_metadata: list[dict] = []
    chunk_count = 0

    for index, opinion in enumerate(opinions):
        text = extract_text(opinion)
        if not text:
            print(f"  [skip] opinion {index} has no extractable text.")
            continue

        chunks = char_chunk(text)
        for chunk_index, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"op_{index}_chunk_{chunk_index}")
            all_metadata.append(
                {
                    "case_name": opinion.get("case_name", f"Opinion_{index}")[:200],
                    "citation": str(opinion.get("citation", ""))[:100],
                    "court": str(opinion.get("court", ""))[:100],
                    "date_filed": str(opinion.get("date_filed", "")),
                    "url": f"https://www.courtlistener.com{opinion.get('absolute_url', '')}",
                    "chunk_index": chunk_index,
                    "total_chunks": len(chunks),
                },
            )
            chunk_count += 1

        processed_file = PROCESSED_DIR / f"opinion_{index}.txt"
        processed_file.write_text(text[:50000], encoding="utf-8")

        if (index + 1) % 10 == 0:
            print(f"  processed {index + 1}/{len(opinions)} opinions, {chunk_count} chunks so far...")

    print(f"[build_index] Embedding {len(all_chunks)} chunks...")
    batch_size = 64
    for start in range(0, len(all_chunks), batch_size):
        batch_texts = all_chunks[start : start + batch_size]
        batch_ids = all_ids[start : start + batch_size]
        batch_metas = all_metadata[start : start + batch_size]
        embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()
        collection.add(
            documents=batch_texts,
            embeddings=embeddings,
            ids=batch_ids,
            metadatas=batch_metas,
        )
        print(f"  indexed {min(start + batch_size, len(all_chunks))}/{len(all_chunks)} chunks...")

    print(f"[build_index] Done. {collection.count()} chunks stored in ChromaDB at {CHROMA_PATH}")


def main() -> None:
    opinions = fetch_opinions()
    build_chromadb_index(opinions)


if __name__ == "__main__":
    main()
