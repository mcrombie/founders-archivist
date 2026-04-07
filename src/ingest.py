import json
import re
from pathlib import Path
from typing import Any

from datetime import datetime

from document_schema import HistoricalDocument


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
OUTPUT_FILE = PROCESSED_DIR / "documents.json"


import re
from datetime import datetime
from typing import Any


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


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

def extract_surname(name: str) -> str:
    """
    Handles:
    - 'Madison, James' -> 'madison'
    - 'James Madison' -> 'madison'
    - '' -> 'unknown'
    """
    name = name.strip()
    if not name:
        return "unknown"

    if "," in name:
        surname = name.split(",", 1)[0].strip()
    else:
        parts = name.split()
        surname = parts[-1] if parts else "unknown"

    return slugify(surname) or "unknown"


def short_title_key(title: str) -> str:
    """
    Creates a short, readable title token for non-letter docs.
    """
    title_slug = slugify(title)

    replacements = {
        "the-federalist-number-10-22-november-1787": "federalist-10",
        "the-federalist-number-10": "federalist-10",
    }

    if title_slug in replacements:
        return replacements[title_slug]

    words = [w for w in title_slug.split("-") if w not in {"the", "a", "an"}]
    return "-".join(words[:4]) if words else "untitled"


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip()

    paragraphs = [re.sub(r"[ \t]+", " ", p).strip() for p in text.split("\n\n")]
    paragraphs = [p for p in paragraphs if p]

    return "\n\n".join(paragraphs)


def build_doc_id(author: str, date: str, recipient: str, title: str, document_type: str) -> str:
    author_key = extract_surname(author)
    date_key = date.replace("-", "_") if date else "unknown_date"

    if recipient.strip():
        recipient_key = extract_surname(recipient)
        return f"{author_key}_{date_key}_{recipient_key}"

    title_key = short_title_key(title)
    return f"{author_key}_{date_key}_{title_key}"


def normalize_doc(data: dict[str, Any]) -> HistoricalDocument:
    title = data["title"].strip()
    author = data["author"].strip()
    recipient = data.get("recipient", "").strip()

    raw_date = data["date"].strip()
    parsed_date = None
    for fmt in ("%Y-%m-%d", "%d %B %Y", "%d %b. %Y", "%d %b %Y"):
        try:
            parsed_date = datetime.strptime(raw_date, fmt)
            break
        except ValueError:
            continue

    if parsed_date is None:
        raise ValueError(f"Unsupported date format: {raw_date}")

    date = parsed_date.strftime("%Y-%m-%d")
    year = parsed_date.year

    document_type = data["document_type"].strip().lower()
    source_url = data["source_url"].strip()
    text = clean_text(data["text"])
    collection = data.get("collection", "Founders Online").strip()
    tags = [str(tag).strip().lower() for tag in data.get("tags", []) if str(tag).strip()]

    doc_id = build_doc_id(author, date, recipient, title, document_type)

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