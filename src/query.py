import os
from typing import Any

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "founders_chunks"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


def get_collection():
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(name=COLLECTION_NAME)


def embed_query(client: OpenAI, query: str) -> list[float]:
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=query,
    )
    return response.data[0].embedding


def retrieve(collection, query_embedding, k=5):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas"],
    )
    return results


def build_context(results) -> str:
    contexts = []
    for i in range(len(results["documents"][0])):
        text = results["documents"][0][i]
        meta = results["metadatas"][0][i]

        context_block = (
            f"[Source {i+1}]\n"
            f"Title: {meta['title']}\n"
            f"Author: {meta['author']}\n"
            f"Date: {meta['date']}\n\n"
            f"{text}\n"
        )
        contexts.append(context_block)

    return "\n\n---\n\n".join(contexts)


def generate_answer(client: OpenAI, query: str, context: str) -> str:
    prompt = f"""
You are a historical research assistant.

Answer the question using ONLY the provided sources.
Cite sources like [Source 1], [Source 2].

If the answer is not clearly supported, say so.

Question:
{query}

Sources:
{context}
"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.query \"your question here\"")
        return

    query = " ".join(sys.argv[1:])

    openai_client = get_openai_client()
    collection = get_collection()

    print("Embedding query...")
    query_embedding = embed_query(openai_client, query)

    print("Retrieving relevant chunks...")
    results = retrieve(collection, query_embedding, k=5)

    context = build_context(results)

    print("\nGenerating answer...\n")
    answer = generate_answer(openai_client, query, context)

    print("ANSWER:\n")
    print(answer)

    print("\nSOURCES:\n")
    for i, meta in enumerate(results["metadatas"][0], start=1):
        print(f"[{i}] {meta['title']} ({meta['date']})")
        print(f"    {meta['source_url']}\n")


if __name__ == "__main__":
    main()