# Ingestion Pipeline (`src/ingestion`)

The `ingestion` package implements **Pipeline 1** of the architecture. It is responsible for discovering raw PDF documents, extracting high-fidelity page text across multiple backends, segmenting text into sentence-aware and section-preserving chunks, managing parameter-sensitive cryptographic disk caches, and populating storage backends.

---

## 📂 Module Breakdown

```
src/ingestion/
├── __init__.py      # Exports IngestionPipeline, chunkers, extractors, storage, cache
├── extractors.py    # Multi-backend PDF text extraction (pypdfium2 -> pdfplumber -> pypdf)
├── chunkers.py      # Sentence-aware & active section-header preserving chunking
├── cache.py         # Parameter- & mtime-sensitive SHA-256 disk caching
├── storage.py       # Decoupled chunk storage abstractions (InMemoryChunkStore, BaseChunkStore)
└── pipeline.py      # IngestionPipeline orchestrator coordinating extraction, chunking, and persistence
```

---

## ⚙️ Key Architectural Features

### 1. Resilient Multi-Backend PDF Extraction (`extractors.py`)
PDF text extraction gracefully cascades across three high-performance Python PDF engines:
1. **`pypdfium2`** (Primary): C++ PDFium bindings; provides clean character extraction without hanging or font corruption. Cleans special byte artifacts (`\ufffe`, `\xad`).
2. **`pdfplumber`** (Secondary): Layout-aware extraction used if PDFium is unavailable.
3. **`pypdf` / `PyPDF2`** (Fallback): Pure-Python standard extraction.

Includes `iter_pdf_documents(corpus_dir)` — a generator yielding `(pdf_filename, [(page_num, text), ...])` suitable for memory-efficient streaming and big-data processing.

### 2. Structured & Sentence-Aware Chunking (`chunkers.py`)
Unlike naive fixed-character chunkers that slice sentences in half or lose context, this chunker:
- **Respects Sentence Boundaries**: Uses `split_into_sentences()` via punctuation boundary regex `(?<=[.?!])\s+(?=[A-Z0-9"“])` and paragraph separation.
- **Tracks Active Section Headers**: Employs `SECTION_HEADER_REGEX` to detect section headings (e.g., `Abstract`, `Introduction`, `§ 3 The Binding Constraint Thesis`) and propagates them as metadata to every chunk in that section.
- **Window Overlap**: Configured with `max_words=120`, `overlap_sentences=1`, and `min_chunk_words=25`.

### 3. Cryptographic State & Parameter Caching (`cache.py`)
Avoids expensive re-extraction on startup using a parameter- and content-sensitive SHA-256 fingerprint:
```python
cache_key = compute_cache_key(corpus_dir, max_words=120, overlap_sentences=1, extra_tag="")
# Output: 16-character hex hash, e.g. "4c017b2532fe4e70"
```
The hash incorporates:
- The absolute corpus directory path
- Chunking configuration (`max_words`, `overlap_sentences`, model tag)
- Per-file metadata for every PDF (`filename`, `mtime`, `size`)

**Benefit**: Adding, editing, or deleting a PDF automatically invalidates the cache; untouched corpora load in ~0.02 seconds.

### 4. Decoupled Storage Layer (`storage.py`)
- **`BaseChunkStore`**: Abstract interface defining `add_chunks()`, `get_chunk()`, `get_all_chunks()`, `get_texts()`.
- **`InMemoryChunkStore`**: High-performance in-memory implementation providing $O(1)$ ID lookups and contiguous text slicing.
- **Big-Data Readiness**: Can be swapped with Parquet, SQLite, BigQuery, or vector database backends (e.g. LanceDB, Qdrant) without changing retrieval logic.

---

## 🚀 Execution & Usage

### Via CLI:
```bash
# Ingest all PDFs in corpus/ and cache results
python -m src.cli ingest --corpus corpus

# Force rebuild cache even if valid
python -m src.cli ingest --corpus corpus --force
```

### Via Python API:
```python
from src.ingestion import IngestionPipeline

# Run pipeline
pipeline = IngestionPipeline(corpus_dir="corpus/", max_words=120, overlap_sentences=1)
chunk_store, total_pages, pdf_paths = pipeline.run()

print(f"Ingested {len(pdf_paths)} PDFs ({total_pages} pages) into {len(chunk_store)} structured chunks.")

# Retrieve a specific chunk
chunk = chunk_store.get_chunk(1561)
print(chunk.to_formatted_text())
```
