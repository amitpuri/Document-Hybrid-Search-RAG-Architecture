# Ingestion Pipeline (`src/ingestion`)

The `ingestion` package implements **Pipeline 1** of the architecture. It is responsible for discovering raw PDF documents, extracting high-fidelity text across multiple parser backends, segmenting text into sentence-aware and section-preserving chunks, extracting domain entities and relations to construct a persistent Knowledge Graph, enforcing parameter-regime consistency across incremental runs, managing partitioned Apache Parquet datasets, and populating storage backends.

---

## 📂 Module Breakdown

```
src/ingestion/
├── __init__.py          # Exports IngestionPipeline, chunkers, extractors, storage, cache, graph
├── extractors.py        # Multi-backend PDF text extraction (pypdfium2 -> pdfplumber -> pypdf)
├── chunkers.py          # Sentence-aware & active section-header preserving chunking
├── graph_extractor.py   # Domain entity & typed relation extraction with IDF-weighted graph building
├── graph_store.py       # NetworkX knowledge graph storage, 1-hop traversal & community detection
├── arxiv_fetcher.py     # Rate-limited arXiv API client for automated corpus growth (Phase 4)
├── cache.py             # Regime hashing, dataset manifest management, atomic Parquet writer
├── storage.py           # Decoupled chunk storage abstractions (BaseChunkStore, InMemory, Parquet, Qdrant)
└── pipeline.py          # IngestionPipeline orchestrator coordinating extraction, chunking, graph, and persistence
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
- **Tracks Active Section Headers**: Employs `SECTION_HEADER_REGEX` to detect section headings (e.g., `Abstract`, `Introduction`, `§ 3 The Binding Constraint Thesis`) and dynamically propagates them as metadata to every chunk in that section.
- **Window Overlap**: Configured with `max_words=200`, `overlap_sentences=1`, and `min_chunk_words=25`.

### 3. Knowledge Graph Construction (`graph_extractor.py`, `graph_store.py`)
Builds an explicit entity-relation knowledge graph alongside the chunk index:
- **Domain Entity Extraction**: Extracts technical acronyms (*"POMDP"*, *"RRF"*, *"MMR"*), capitalized compound terms (*"Binding Constraint Thesis"*, *"Dynamic Tiered AgentRunner"*), and key architectural components.
- **Typed Relation Extraction**: Discovers explicit semantic relationships (`DEFINES`, `IMPLEMENTS`, `PART_OF`, `AFFILIATED_WITH`) and sentence-level `CO_OCCURS_WITH` associations.
- **Corpus-Level IDF Weighting**: Scales entity activations by inverse document frequency, preventing ubiquitous generic terms (*"model"*, *"agent"*, *"system"*) from dominating coined domain concepts.
- **Persistent NetworkX Graph Index (`graph_store.py`)**: Serialized to disk cache, supporting 1-hop local neighborhood traversal, degree/IDF activation scoring, and Louvain community detection fallback.

### 4. Decoupled Storage Layer & Storage Backends (`storage.py`)
The storage layer decouples retrieval logic from physical persistence through interchangeable backends:
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
  - **Metadata Loss Bug Fix**: Serializes custom chunk metadata to `metadata_json` with complete fidelity upon reload.
  - **Column Projection Pushdown**: `get_texts()` loads *only* the `formatted_text` column via `scanner(columns=["formatted_text"])`, bypassing full chunk text and metadata to reduce I/O and RAM overhead by ~70% during retriever indexing.
  - **Predicate Pushdown & Partition Pruning**: `filter()` evaluates PyArrow compute kernels (`pc.equal`, `pc.match_substring`) and leverages native row-group min/max statistics to skip non-matching partition files.
  - **Zero-Dict Direct Offset Addressing**: Contiguous IDs map directly to row offsets without allocating millions of Python dictionary entries.
- **`QdrantChunkStore`**: Enterprise vector and chunk database backend backed by local or remote [Qdrant](https://qdrant.tech/):
  - **Unified Point & Payload Storage**: Stores chunk texts and structured provenance fields as point payloads (`chunk_id`, `doc_name`, `page_num`, `section`, `text`, `metadata`).
  - **Server-Side Predicate Filtering**: Implements high-speed metadata filtering via native Qdrant `FieldCondition` matches (`KEYWORD` index on `doc_name`, `INTEGER` index on `page_num`, and `TEXT` index on `section`).
  - **Named Vectors Support**: Hosts dense embeddings under the `"dense"` vector namespace alongside text payloads.
  - **Strict Index-Order Guarantees**: Sorts scrolled points by `chunk_id` ascending to ensure zero drift against ground-truth evaluation datasets.

### 5. Partitioned Dataset Topology & Incremental Ingestion (`pipeline.py`, `cache.py`)
Persists partitioned datasets under `.cache/chunks_dataset/`:
```
.cache/chunks_dataset/
├── _manifest.json             # Global metadata, partition catalog, next_chunk_id, regime hash
├── doc=1604.08127v1.pdf/
│   └── data_0000.parquet      # Chunks 0..264
├── doc=2509.01063v1.pdf/
│   └── data_0000.parquet      # Chunks 265..395
└── ...
```

- **Monotonic Insertion-Order `chunk_id` Assignment**: New documents are appended at `next_chunk_id`. Existing chunks in earlier documents never renumber, preserving ground-truth benchmark indices and embeddings.
- **Chunking-Parameter Regime Guard**: Computes `chunking_regime_hash` (`max_words`, `overlap_sentences`, chunker version). Raises `IngestionRegimeMismatchError` if an incremental run uses conflicting chunking parameters, preventing silent boundary drift.
- **Incremental Diffing**: Checks file `mtime` and `size` against `_manifest.json`. Only new or modified PDFs trigger extraction and partition writes; untouched partitions are preserved.
- **Atomic Writes**: Parquet partition files and `_manifest.json` are written to temporary files (`.tmp_{uuid}`) and atomically swapped via `os.replace`.

### 6. arXiv Automated Corpus Growth (`arxiv_fetcher.py`) - Phase 4
Rate-limited arXiv API client for continuous corpus expansion:
- **Rate Limiting**: Enforces 3-second minimum delay between requests with exponential backoff on HTTP 429 errors
- **Category Filtering**: Supports category-based filtering (cs.AI, cs.IR, cs.CL, etc.) for targeted technical domains
- **Append-Only ID Assignment**: Integrates with IngestionPipeline to maintain monotonic chunk IDs across corpus growth
- **CLI Integration**: `python -m src.cli ingest --arxiv --category cs.AI --limit 100`
- **MCP Server**: `src/mcp/arxiv_server.py` provides conversational agent integration for corpus growth

---

## 🐳 Local Qdrant Setup

To run Qdrant locally for vector storage and chunk management:

```bash
# Start local Qdrant container with persistent storage
docker compose up -d
# or via standalone docker run:
docker run -d -p 6333:6333 -p 6334:6334 -v ${PWD}/qdrant_storage:/qdrant/storage:z --name qdrant_local qdrant/qdrant

# Inspect local Qdrant container status & collection statistics
python -m src.cli qdrant-status
```
Local Web Dashboard is accessible at: **[http://localhost:6333/dashboard](http://localhost:6333/dashboard)**

---

## 🚀 Execution & Usage

### Via CLI:
```bash
# Ingest all PDFs in corpus/ into Parquet dataset (default)
python -m src.cli ingest --corpus corpus

# Ingest all PDFs into local Qdrant vector database
python -m src.cli ingest --corpus corpus --storage qdrant

# Force rebuild entire dataset and knowledge graph from scratch
python -m src.cli ingest --corpus corpus --force

# Use lightweight in-memory backend
python -m src.cli ingest --corpus corpus --storage memory

# Ingest arXiv papers with category filtering (Phase 4)
python -m src.cli ingest --arxiv --category cs.AI --limit 100

# Ingest arXiv papers with custom query
python -m src.cli ingest --arxiv --arxiv-query "reinforcement learning" --limit 50

# Ingest arXiv papers from multiple categories
python -m src.cli ingest --arxiv --category cs.AI,cs.IR,cs.CL --limit 200
```

### Via Python API:
```python
from src.ingestion import IngestionPipeline

# 1. Run ingestion pipeline (extracts text, chunks, builds KG and Parquet store)
pipeline = IngestionPipeline(
    corpus_dir="corpus/",
    max_words=200,
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

# 5. Ingestion with Qdrant vector database backend
qdrant_pipeline = IngestionPipeline(
    corpus_dir="corpus/",
    storage_backend="qdrant"
)
qdrant_store, _, _ = qdrant_pipeline.run()
filtered_chunks = qdrant_store.filter(doc_name="2605.23950v1.pdf")
print(f"Qdrant filtered {len(filtered_chunks)} chunks using server-side field conditions.")
```
