
---

# ROADMAP.md (use this)

Keep this tight and actionable.

```markdown
# Roadmap

## Current State

Working CLI RAG system over a small historical database.

Core pipeline:
ingest → chunk → embed → query

---

## Priority Improvements

### 1. Retrieval Quality

- Deduplicate results by `doc_id`
- Ensure diversity across documents
- Improve relevance ranking

---

### 2. Chunk Quality

- Refine overlap behavior
- Improve handling of small trailing chunks
- Further tune sentence-based splitting

---

### 3. Text Cleaning

- Remove inline editorial note markers safely
- Preserve historical meaning while reducing noise

---

### 4. Metadata Filtering

- Filter by:
  - author
  - date range
  - document type

Example:
"Madison on factions after 1787"

---

### 5. Ingestion Improvements

- Semi-automated extraction from Founders Online
- HTML parsing for structured fields
- Batch ingestion with rate limiting

---

### 6. CLI Improvements

- Better output formatting
- Optional display of excerpts
- Adjustable number of sources
- Debug mode for retrieval inspection

---

## Medium-Term Goals

### Hybrid Retrieval

- Combine semantic search with keyword search

### Reranking

- Re-rank retrieved chunks for higher precision

### Corpus Expansion

- Add:
  - Jefferson
  - Hamilton
  - Adams correspondence

---

## Long-Term Ideas

- Web UI for browsing and querying
- Document-level summaries
- Timeline-based navigation
- Cross-document thematic linking
- Integration with other historical datasets

---

## Ownership Direction

- Current focus: working prototype
- Next phase: improve retrieval + ingestion

This project is intended to evolve collaboratively.