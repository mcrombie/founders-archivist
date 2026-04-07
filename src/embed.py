import json
import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

CHUNKS_FILE = Path("data/processed/chunks.json")
CHROMA_DIR = Path("data/chroma")
COLLECTION_NAME = "founders_chunks"
EMBED_MODEL = "text-embedding-3-small"


def load_chunks() -> list[dict[str, Any]]:
    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in the environment.")
    return OpenAI(api_key=api_key)


def get_chroma_collection():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection


def chunk_to_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    """
    Keep metadata flat and Chroma-safe.
    """
    return {
        "chunk_id": chunk["chunk_id"],
        "doc_id": chunk["doc_id"],
        "chunk_index": int(chunk["chunk_index"]),
        "title": chunk["title"],
        "author": chunk["author"],
        "recipient": chunk.get("recipient", ""),
        "date": chunk["date"],
        "year": int(chunk["year"]),
        "collection": chunk["collection"],
        "document_type": chunk["document_type"],
        "source_url": chunk["source_url"],
        "word_count": int(chunk["word_count"]),
        "tags": ", ".join(chunk.get("tags", [])),
    }


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def batchify(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def main() -> None:
    chunks = load_chunks()
    openai_client = get_openai_client()
    collection = get_chroma_collection()

    batch_size = 20

    chunk_batches = batchify(chunks, batch_size)

    total_added = 0

    for batch_num, batch in enumerate(chunk_batches, start=1):
        ids = [chunk["chunk_id"] for chunk in batch]
        texts = [chunk["text"] for chunk in batch]
        metadatas = [chunk_to_metadata(chunk) for chunk in batch]

        print(f"Embedding batch {batch_num}/{len(chunk_batches)} ({len(batch)} chunks)...")
        embeddings = embed_texts(openai_client, texts)

        collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        total_added += len(batch)

    print(f"\nStored {total_added} chunks in Chroma collection '{COLLECTION_NAME}' at {CHROMA_DIR}")


if __name__ == "__main__":
    main()