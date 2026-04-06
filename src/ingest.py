import json
import re
from pathlib import Path
from typing import Any

from document_schema import HistoricalDocument


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
OUTPUT_FILE = PROCESSED_DIR / "documents.json"


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def build_doc_id(author: str, date: str, recipient: str, title: str) -> str:
    author_slug = slugify(author) if author else "unknown-author"
    recipient_slug = slugify(recipient) if recipient else "unknown-recipient"
    date_slug = date if date else "unknown-date"
    title_slug = slugify(title)[:40] if title else "untitled"
    return f"{author_slug}-{date_slug}-{recipient_slug}-{title_slug}"


def load_raw_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_raw_doc(data: dict[str, Any], path: Path) -> None:
    required_fields = [
        "title",
        "author",
        "date",
        "document_type",
        "source_url",
        "text",
    ]
    missing = [field for field in required_fields if not data.get(field)]
    if missing:
        raise ValueError(f"{path.name} is missing required fields: {missing}")


def normalize_doc(data: dict[str, Any]) -> HistoricalDocument:
    title = data["title"].strip()
    author = data["author"].strip()
    recipient = data.get("recipient", "").strip()
    date = data["date"].strip()
    year = int(date[:4])
    document_type = data["document_type"].strip().lower()
    source_url = data["source_url"].strip()
    text = data["text"].strip()
    collection = data.get("collection", "Founders Online").strip()
    tags = data.get("tags", [])

    doc_id = build_doc_id(author, date, recipient, title)

    return HistoricalDocument(
        doc_id=doc_id,
        title=title,
        author=author,
        recipient=recipient,
        date=date,
        year=year,
        collection=collection,
        document_type=document_type,
        source_url=source_url,
        text=text,
        tags=tags,
    )


def ingest_documents() -> list[dict]:
    documents = []

    for path in sorted(RAW_DIR.glob("*.json")):
        raw = load_raw_json(path)
        validate_raw_doc(raw, path)
        normalized = normalize_doc(raw)
        documents.append(normalized.to_dict())
        print(f"Ingested: {path.name} -> {normalized.doc_id}")

    return documents


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    documents = ingest_documents()

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(documents)} documents to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()