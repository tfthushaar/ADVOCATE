"""
build_index.py
Fetches employment wrongful termination opinions from CourtListener,
chunks them into 512-token segments with 64-token overlap, embeds them
with all-MiniLM-L6-v2, and stores them in a local ChromaDB instance.
"""

import os
import json
import time
import requests
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("COURTLISTENER_BASE_URL", "https://www.courtlistener.com/api/rest/v4")
CHROMA_PATH = os.getenv("CHROMA_PERSIST_PATH", "./advocate/data/chroma_db")
RAW_DIR = Path("./advocate/data/raw_cases")
PROCESSED_DIR = Path("./advocate/data/processed_cases")
CHUNK_SIZE = 512       # tokens (approx characters / 4)
CHUNK_OVERLAP = 64
COLLECTION_NAME = "employment_law"
TARGET_OPINIONS = 70


def fetch_opinions(max_results: int = TARGET_OPINIONS) -> list[dict]:
    """Pull wrongful termination opinions from federal circuits (2010-2024)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = RAW_DIR / "opinions_cache.json"

    if cache_file.exists():
        print(f"[build_index] Loading {max_results} opinions from cache.")
        with open(cache_file) as f:
            opinions = json.load(f)
        return opinions[:max_results]

    print("[build_index] Fetching opinions from CourtListener …")
    opinions = []
    url = f"{BASE_URL}/search/"
    params = {
        "q": "wrongful termination employment",
        "type": "o",
        "order_by": "score desc",
    }
    token = os.getenv("COURTLISTENER_API_TOKEN", "")
    headers = {"User-Agent": "ADVOCATE-Research-Tool/1.0"}
    if token:
        headers["Authorization"] = f"Token {token}"
    else:
        print(
            "\n[build_index] WARNING: COURTLISTENER_API_TOKEN not set.\n"
            "  CourtListener REST API v4 requires a free account token.\n"
            "  1. Register at https://www.courtlistener.com/sign-in/\n"
            "  2. Get your token at https://www.courtlistener.com/api/rest/v4/\n"
            "  3. Add COURTLISTENER_API_TOKEN=<token> to your .env file\n"
        )

    # First fetch search results to find opinion IDs
    search_results = []
    while url and len(search_results) < max_results:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        if resp.status_code == 401:
            raise RuntimeError(
                "CourtListener returned 401 Unauthorized.\n"
                "Register for a free account at https://www.courtlistener.com/sign-in/\n"
                "then add COURTLISTENER_API_TOKEN=<your_token> to your .env file."
            )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        search_results.extend(results)
        print(f"  found {len(search_results)} cases so far …")
        url = data.get("next")
        params = {}
        time.sleep(1)
        
    search_results = search_results[:max_results]
    
    # Now fetch the actual text for each found opinion
    print(f"[build_index] Fetching full text for {len(search_results)} cases …")
    for sr in search_results:
        op_list = sr.get("opinions", [])
        if not op_list:
            continue
        op_id = op_list[0].get("id")
        
        # Merge search metadata with fetched opinion body
        op_dict = {
            "case_name": sr.get("caseName", ""),
            "citation": sr.get("citation", [""])[0] if isinstance(sr.get("citation"), list) and sr.get("citation") else sr.get("citation", ""),
            "court": sr.get("court", ""),
            "date_filed": sr.get("dateFiled", ""),
            "absolute_url": sr.get("absolute_url", ""),
        }
        
        try:
            op_resp = requests.get(f"{BASE_URL}/opinions/{op_id}/", headers=headers, timeout=10)
            if op_resp.status_code == 200:
                op_data = op_resp.json()
                op_dict["plain_text"] = op_data.get("plain_text", "")
                op_dict["html_with_citations"] = op_data.get("html_with_citations", "")
                op_dict["html"] = op_data.get("html", "")
                opinions.append(op_dict)
            time.sleep(0.5)
        except Exception as e:
            print(f"  [warn] Failed to fetch opinion {op_id}: {e}")

    with open(cache_file, "w") as f:
        json.dump(opinions, f, indent=2)
    print(f"[build_index] Saved {len(opinions)} raw opinions to cache.")
    return opinions


def extract_text(opinion: dict) -> str:
    """Pull plain text from various opinion text fields."""
    for field in ("plain_text", "html_with_citations", "html", "xml_harvard"):
        text = opinion.get(field, "")
        if text and len(text) > 100:
            # Strip basic HTML tags if present
            import re
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text
    # Fallback: fetch the opinion detail URL for the text
    return ""


def char_chunk(text: str, chunk_chars: int = CHUNK_SIZE * 4,
               overlap_chars: int = CHUNK_OVERLAP * 4) -> list[str]:
    """Split text into overlapping character-level chunks (proxy for token chunks)."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_chars - overlap_chars
    return chunks


def build_chromadb_index(opinions: list[dict]) -> None:
    """Embed and store all opinion chunks in ChromaDB."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)

    print("[build_index] Loading embedding model …")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    # Drop existing collection to allow rebuilds
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    all_chunks, all_ids, all_metas = [], [], []
    chunk_id = 0

    for i, op in enumerate(opinions):
        text = extract_text(op)
        if not text:
            print(f"  [skip] opinion {i} has no extractable text.")
            continue

        case_name = op.get("case_name", f"Opinion_{i}")
        citation = op.get("citation", "")
        court = op.get("court", "")
        date_filed = op.get("date_filed", "")
        absolute_url = op.get("absolute_url", "")

        chunks = char_chunk(text)
        for j, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"op_{i}_chunk_{j}")
            all_metas.append({
                "case_name": case_name[:200],
                "citation": str(citation)[:100],
                "court": str(court)[:100],
                "date_filed": str(date_filed),
                "url": f"https://www.courtlistener.com{absolute_url}",
                "chunk_index": j,
                "total_chunks": len(chunks),
            })
            chunk_id += 1

        # Save processed text
        proc_file = PROCESSED_DIR / f"opinion_{i}.txt"
        proc_file.write_text(text[:50000], encoding="utf-8")

        if (i + 1) % 10 == 0:
            print(f"  processed {i+1}/{len(opinions)} opinions, {chunk_id} chunks so far …")

    print(f"[build_index] Embedding {len(all_chunks)} chunks …")
    batch_size = 64
    for start in range(0, len(all_chunks), batch_size):
        batch_texts = all_chunks[start:start + batch_size]
        batch_ids = all_ids[start:start + batch_size]
        batch_metas = all_metas[start:start + batch_size]
        embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()
        collection.add(
            documents=batch_texts,
            embeddings=embeddings,
            ids=batch_ids,
            metadatas=batch_metas,
        )
        print(f"  indexed {min(start + batch_size, len(all_chunks))}/{len(all_chunks)} chunks …")

    print(f"[build_index] Done. {collection.count()} chunks stored in ChromaDB at {CHROMA_PATH}")


def main():
    opinions = fetch_opinions()
    build_chromadb_index(opinions)


if __name__ == "__main__":
    main()
