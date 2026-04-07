# Founders Archivist

A CLI-based historical research assistant built on Retrieval-Augmented Generation (RAG), using primary source documents from Founders Online.

This project demonstrates how AI can be used to navigate and synthesize historical texts while preserving direct citation to original sources.

---

## Current Features

- Ingests manually curated primary source documents
- Normalizes metadata (author, date, recipient, etc.)
- Splits documents into paragraph-aware chunks
- Embeds chunks using OpenAI embeddings
- Stores embeddings in a local Chroma vector database
- Supports CLI queries over the corpus
- Returns:
  - grounded answers
  - cited sources
  - relevant document excerpts

---

## Example Query

```bash
python src/query.py "What did Madison argue about factions?"

Embedding query...
Retrieving relevant chunks...

Generating answer...

ANSWER:

James Madison argued that factions are groups of citizens united by a common interest or passion that is adverse to the rights of other citizens or the overall interests of the community. He identified two methods to address the issues caused by factions: removing their causes or controlling their effects. However, he deemed both methods problematic. Removing the causes would require either destroying liberty, which is essential for political life, or ensuring that all citizens have the same opinions and interests, which he considered impractical due to the inherent diversity of human nature and interests.

Madison emphasized that the latent causes of faction are rooted in human nature and the unequal distribution of property, which leads to different interests and parties within society. He argued that a pure democracy is inadequate for controlling factions because a majority could easily oppress the minority. Instead, he advocated for a republic, where representatives could refine and enlarge public views, thus better controlling the effects of factions and protecting the rights of all citizens [Source 1], [Source 2], [Source 3], [Source 4].


    https://founders.archives.gov/documents/Madison/01-10-02-0178

[2] The Federalist Number 10, [22 November] 1787 (1787-11-22)
    https://founders.archives.gov/documents/Madison/01-10-02-0178

[3] The Federalist Number 10, [22 November] 1787 (1787-11-22)
    https://founders.archives.gov/documents/Madison/01-10-02-0178

[4] The Federalist Number 10, [22 November] 1787 (1787-11-22)
    https://founders.archives.gov/documents/Madison/01-10-02-0178

[5] James Madison to Thomas Jefferson, 24 October 1787 (1787-10-24)
    https://founders.archives.gov/documents/Madison/01-10-02-0151

---


## Project Structure

data/
  raw/          # manually created source documents
  processed/    # normalized documents + chunks
  chroma/       # vector database

src/
  ingest.py     # normalize raw documents
  chunk.py      # split documents into chunks
  embed.py      # create embeddings + store in Chroma
  query.py      # CLI query interface

  ---

Setup

1. Install dependencies
pip install -r requirements.txt
2. Set environment variables

Create a .env file:

OPENAI_API_KEY=your_api_key_here

 ---


Pipeline

Run these steps in order:

python src/ingest.py
python src/chunk.py
python src/embed.py

Then query:

python src/query.py "your question here"

 ---


Current Scope

Small, manually curated dataset (~3 documents, ~20 chunks)
Focused on James Madison and Federalist-era texts
CLI-only interface (no UI)

 ---


Limitations

Retrieval currently returns multiple chunks from the same document
Inline editorial note markers are still present in text
Chunking is paragraph-based with simple fallback splitting
No filtering, reranking, or metadata-aware querying yet
Ingestion is manual (no automated extraction)

---


Purpose

This is a proof-of-concept system for:
    historical research assistance
    primary source navigation
    RAG system design on structured archival material

 ---


Data Source

Documents sourced from Founders Online (National Archives).

Note:

    Primary texts are public domain
    Editorial annotations are excluded

 ---


Next Steps

See ROADMAP.md