# Ingestion Pipeline (`src/ingestion`)

The `ingestion` package implements **Pipeline 1** of the architecture. It is responsible for discovering raw PDF documents, extracting high-fidelity page text across multiple backends, segmenting text into sentence-aware and section-preserving chunks, enforcing parameter-regime consistency across incremental runs, managing partitioned Apache Parquet datasets, and populating storage backends.

---

## 📂 Module Breakdown

```
src/ingestion/
├── __init__.py      # Exports IngestionPipeline, chunkers, extractors, storage, cache, schema
├── extractors.py    # Multi-backend PDF text extraction (pypdfium2 -> pdfplumber -> pypdf)
├── chunkers.py      # Sentence-aware & active section-header preserving chunking
├── cache.py         # Regime hashing, dataset manifest management, atomic Parquet writer
├── storage.py       # Decoupled chunk storage abstractions (BaseChunkStore, InMemory, Parquet)
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

### 3. Decoupled Storage Layer & Big-Data Parquet Datasets (`storage.py`)
The storage layer decouples retrieval logic from physical persistence:
- **`BaseChunkStore`**: Abstract interface defining:
  - `add_chunks()`, `get_chunk()`, `get_all_chunks()`, `get_texts()`, `__len__()`
  - Concrete defaults for `get_chunks(chunk_ids)` (batch lookup) and `filter(doc_name, page_num, section)`
- **`InMemoryChunkStore`**: High-performance in-memory array and hashmap storage for testing and lightweight runs.
- **`ParquetChunkStore`**: Big-data columnar storage backed by `pyarrow.dataset`:
  - **Strict PyArrow Schema**:
    ```python
    CHUNK_PYARROW_SCHEMA = pa.schema([
        pa.field("chunk_id", pa.int64(), nullable=False),
        pa.field("doc_name", pa.string(), nullable=False),
        pa.field("page_num", pa.int32(), nullable=False),
        pa.field("section", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("formatted_text", pa.string(), nullable=False),
        pa.field("metadata_json", pa.string(), nullable=True),
    ])
    ```
  - **Metadata Loss Bug Fix**: Serializes custom chunk metadata to `metadata_json` with complete fidelity upon reload (curing the legacy string-parsing loss bug).
  - **Column Projection Pushdown**: `get_texts()` loads *only* the `formatted_text` column via `scanner(columns=["formatted_text"])`, bypassing full chunk text and metadata to reduce I/O and RAM overhead by ~70% during retriever indexing.
  - **Predicate Pushdown & Partition Pruning**: `filter()` evaluates PyArrow compute kernels (`pc.equal`, `pc.match_substring`) and leverages native row-group min/max statistics to skip non-matching partition files.
  - **Zero-Dict Direct Offset Addressing**: Monotonic ascending contiguous IDs map directly to row offsets without allocating millions of Python dictionary entries.
  - **Automatic Dual-Mode Access**: Automatically uses zero-copy in-memory tables for small corpora (< 50,000 chunks) and out-of-core memory-mapped scanners for larger corpora.

### 4. Partitioned Dataset Topology & Incremental Ingestion (`pipeline.py`, `cache.py`)
Instead of monolithic files that re-chunk everything on any modification, the pipeline persists partitioned datasets under `.cache/chunks_dataset/`:
```
.cache/chunks_dataset/
├── _manifest.json             # Global metadata, partition catalog, next_chunk_id, regime hash
├── doc=1604.08127v1.pdf/
│   └── data_0000.parquet      # Chunks 0..264
├── doc=2509.01063v1.pdf/
│   └── data_0000.parquet      # Chunks 265..395
└── ...
```

- **Monotonic Insertion-Order `chunk_id` Assignment**: New documents are appended at `next_chunk_id`. Existing chunks in earlier documents never renumber, preserving ground-truth benchmark indices, embeddings, and external citations.
- **Chunking-Parameter Regime Guard**: Computes `chunking_regime_hash` (`max_words`, `overlap_sentences`, chunker version). Raises `IngestionRegimeMismatchError` if an incremental run uses conflicting chunking parameters, preventing silent boundary drift.
- **Incremental Diffing**: Checks file `mtime` and `size` against `_manifest.json`. Only new or modified PDFs trigger extraction and partition writes; untouched partitions are preserved.
- **Atomic Writes**: Parquet partition files and `_manifest.json` are written to temporary files (`.tmp_{uuid}`) and atomically swapped via `os.replace`.

---

## 🚀 Execution & Usage

### Via CLI:
```bash
# Ingest all PDFs in corpus/ into Parquet dataset (default)
python -m src.cli ingest --corpus corpus

# Force rebuild entire dataset from scratch
python -m src.cli ingest --corpus corpus --force

# Use legacy in-memory backend
python -m src.cli ingest --corpus corpus --storage memory
```

### Via Python API:
```python
from src.ingestion import IngestionPipeline, ParquetChunkStore

# 1. Run pipeline (uses Parquet storage by default)
pipeline = IngestionPipeline(
    corpus_dir="corpus/",
    max_words=120,
    overlap_sentences=1,
    storage_backend="parquet"
)
chunk_store, total_pages, pdf_paths = pipeline.run()

print(f"Ingested {len(pdf_paths)} PDFs ({total_pages} pages) into {len(chunk_store)} structured chunks.")

# 2. Retrieve a specific chunk (O(1) direct offset lookup)
chunk = chunk_store.get_chunk(1561)
print(chunk.to_formatted_text())

# 3. Predicate filtering (evaluated via PyArrow C++ compute expressions)
pomdp_chunks = chunk_store.filter(doc_name="1604.08127v1.pdf", section="POMDP")
print(f"Found {len(pomdp_chunks)} matching chunks.")

# 4. Fast columnar projection
formatted_texts = chunk_store.get_texts()
print(f"Loaded {len(formatted_texts)} texts via zero-copy column projection.")
```
