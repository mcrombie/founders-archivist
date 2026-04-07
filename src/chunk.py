import json
import re
from pathlib import Path
from typing import Any


INPUT_FILE = Path("data/processed/documents.json")
OUTPUT_FILE = Path("data/processed/chunks.json")


TARGET_WORDS = 700
MAX_WORDS = 900
OVERLAP_PARAGRAPHS = 1


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def split_into_paragraphs(text: str) -> list[str]:
    raw_paragraphs = [p.strip() for p in text.split("\n\n")]
    raw_paragraphs = [p for p in raw_paragraphs if p]

    paragraphs = []
    for paragraph in raw_paragraphs:
        if word_count(paragraph) > MAX_WORDS:
            paragraphs.extend(split_large_paragraph(paragraph, MAX_WORDS))
        else:
            paragraphs.append(paragraph)

    return paragraphs


def split_large_paragraph(paragraph: str, max_words: int) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', paragraph.strip())
    parts = []
    current_sentences = []
    current_words = 0

    for sentence in sentences:
        sentence_words = word_count(sentence)

        if current_sentences and current_words + sentence_words > max_words:
            parts.append(" ".join(current_sentences).strip())
            current_sentences = [sentence]
            current_words = sentence_words
        else:
            current_sentences.append(sentence)
            current_words += sentence_words

    if current_sentences:
        parts.append(" ".join(current_sentences).strip())

    return parts


def chunk_document(doc: dict[str, Any]) -> list[dict[str, Any]]:
    paragraphs = split_into_paragraphs(doc["text"])
    chunks = []

    current_paragraphs: list[str] = []
    current_word_count = 0
    chunk_index = 0
    i = 0

    while i < len(paragraphs):
        paragraph = paragraphs[i]
        paragraph_words = word_count(paragraph)

        # If current chunk exists and adding the next paragraph would exceed MAX_WORDS,
        # flush the current chunk.
        if current_paragraphs and current_word_count + paragraph_words > MAX_WORDS:
            chunk_text = "\n\n".join(current_paragraphs)

            new_chunk = {
                "chunk_id": f"{doc['doc_id']}_chunk_{chunk_index}",
                "doc_id": doc["doc_id"],
                "chunk_index": chunk_index,
                "title": doc["title"],
                "author": doc["author"],
                "recipient": doc["recipient"],
                "date": doc["date"],
                "year": doc["year"],
                "collection": doc["collection"],
                "document_type": doc["document_type"],
                "source_url": doc["source_url"],
                "tags": doc.get("tags", []),
                "text": chunk_text,
                "word_count": current_word_count,
            }

            # Only append if it's not a duplicate of the previous chunk
            if not chunks or chunks[-1]["text"] != new_chunk["text"]:
                chunks.append(new_chunk)


            chunk_index += 1

            overlap = current_paragraphs[-OVERLAP_PARAGRAPHS:] if OVERLAP_PARAGRAPHS > 0 else []
            overlap_word_count = sum(word_count(p) for p in overlap)

            # If overlap + current paragraph would still exceed max, drop overlap.
            if overlap and overlap_word_count + paragraph_words > MAX_WORDS:
                current_paragraphs = []
                current_word_count = 0
            else:
                current_paragraphs = overlap[:]
                current_word_count = overlap_word_count

            continue

        current_paragraphs.append(paragraph)
        current_word_count += paragraph_words
        i += 1

        # Flush once we hit target size
        if current_word_count >= TARGET_WORDS:
            chunk_text = "\n\n".join(current_paragraphs)

            new_chunk = {
                "chunk_id": f"{doc['doc_id']}_chunk_{chunk_index}",
                "doc_id": doc["doc_id"],
                "chunk_index": chunk_index,
                "title": doc["title"],
                "author": doc["author"],
                "recipient": doc["recipient"],
                "date": doc["date"],
                "year": doc["year"],
                "collection": doc["collection"],
                "document_type": doc["document_type"],
                "source_url": doc["source_url"],
                "tags": doc.get("tags", []),
                "text": chunk_text,
                "word_count": current_word_count,
            }

            # Only append if it's not a duplicate of the previous chunk
            if not chunks or chunks[-1]["text"] != new_chunk["text"]:
                chunks.append(new_chunk)

            chunk_index += 1
            overlap = current_paragraphs[-OVERLAP_PARAGRAPHS:] if OVERLAP_PARAGRAPHS > 0 else []
            current_paragraphs = overlap[:]
            current_word_count = sum(word_count(p) for p in current_paragraphs)

    # Flush leftover chunk
    if current_paragraphs:
        chunk_text = "\n\n".join(current_paragraphs)
        new_chunk = {
            "chunk_id": f"{doc['doc_id']}_chunk_{chunk_index}",
            "doc_id": doc["doc_id"],
            "chunk_index": chunk_index,
            "title": doc["title"],
            "author": doc["author"],
            "recipient": doc["recipient"],
            "date": doc["date"],
            "year": doc["year"],
            "collection": doc["collection"],
            "document_type": doc["document_type"],
            "source_url": doc["source_url"],
            "tags": doc.get("tags", []),
            "text": chunk_text,
            "word_count": current_word_count,
        }

        # Only append if it's not a duplicate of the previous chunk
        if not chunks or chunks[-1]["text"] != new_chunk["text"]:
            chunks.append(new_chunk)

    return chunks

def load_documents() -> list[dict[str, Any]]:
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_chunks(chunks: list[dict[str, Any]]) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)


def main() -> None:
    documents = load_documents()
    all_chunks = []

    for doc in documents:
        print(f"Chunking {doc['doc_id']}...")
        doc_chunks = chunk_document(doc)
        all_chunks.extend(doc_chunks)
        print(f"{doc['doc_id']}: {len(doc_chunks)} chunks")

    save_chunks(all_chunks)
    print(f"\nWrote {len(all_chunks)} chunks to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()