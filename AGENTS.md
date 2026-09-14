# AI Agent Operational Guidelines (`AGENTS.md`)

This document serves as the operational manual, behavioral guidelines, and context reference for AI coding agents and automated systems working on the `Document-Hybrid-Search-RAG-Architecture` repository.

---

## 1. Repository Mission & Overview

`Document-Hybrid-Search-RAG-Architecture` is an enterprise-grade information retrieval and RAG codebase designed for scientific, technical, and academic PDF documentation.

The core challenge this repository solves:
> **Why pure vector search fails on technical literature:** Generic dense embeddings (e.g. MiniLM) diffuse technical acronyms (*"StarShell"*, *"POMDP"*, *"AgentRunner"*, *"Looped Flows"*) across generic semantic neighborhoods. BM25 sparse matching provides exact lexical precision, while vector search provides semantic recall. **Hybrid rank fusion (RRF + Deduplication + MMR)** consistently outperforms pure sparse and pure dense systems.

---

## 2. Directory Layout & Organization

```
Document-Hybrid-Search-RAG-Architecture/
├── src/                     # PRODUCTION CODE: Three-Pipeline Architecture
│   ├── common/              # Strongly-typed domain models & text utilities
│   ├── ingestion/           # Pipeline 1: PDF extraction, chunking, caching, storage
│   ├── retrieval/           # Pipeline 2: 13 hybrid search strategies (BM25, TF-IDF, PPMI, Neural, SPECTER2, RRF, MMR)
│   ├── generation/          # Pipeline 3: Context building, RAG prompt templates, LLM adapters
│   ├── evaluation/          # Evaluation suite (MRR, Recall@K, NDCG@5, 14 benchmark queries)
│   ├── engine.py            # High-level HybridSearchEngine binding all 3 pipelines
│   └── cli.py               # Unified CLI interface (search, ask, eval, ingest)
│
├── corpus/                  # 11 Research PDF files (~354 pages, 2,072 structured chunks)
├── .cache/                  # SHA-256 parameter- & state-sensitive disk cache (.pkl & .npy)
├── run_eval.py              # Root convenience script to run the 14-query x 13-strategy benchmark
├── requirements.txt         # Project dependencies
│
├── initial-approach/        # HISTORICAL: Baseline prototype (10 queries x 9 strategies, doc-level eval)
└── improved-approach/       # HISTORICAL: Standalone scripts (14 queries x 12 strategies, chunk-level eval)
```

> [!IMPORTANT]
> - All **new features, enhancements, and production code** must be implemented in `src/`.
> - Do **not** modify `initial-approach/` or `improved-approach/` unless explicitly requested by the user, as they represent frozen developmental milestones.

---

## 3. The Three-Pipeline Architectural Pattern

Agents modifying `src/` must preserve the strict segregation of the three pipelines:

```
Raw PDFs ──> [ Pipeline 1: Ingestion ] ──> Chunk Store
                                                │
Query    ──> [ Pipeline 2: Retrieval ] <────────┘
                    │
            Ranked Chunks (Top-K)
                    │
Query    ──> [ Pipeline 3: Generation ] ──> Grounded Answer with Citations
```

1. **Ingestion Pipeline (`src/ingestion`)**:
   - Must remain decoupled from retrieval algorithms.
   - Preserves sentence boundaries and tracks active section headings via regex.
   - Cache keys must be generated using `compute_cache_key()` incorporating corpus directory path, chunking params, file sizes, and mtimes.
   - Storage abstractions (`BaseChunkStore`, `InMemoryChunkStore`) must be maintained so storage can be swapped for big-data platforms (Parquet/Spark/BigQuery/Vector DBs).

2. **Retrieval Pipeline (`src/retrieval`)**:
   - All 13 strategies must dispatch through `get_strategy_rankings(query)` to maintain identical candidate pools across benchmarks.
   - **Cross-Encoder Architecture Rule**: The cross-encoder must re-rank a *wide, un-deduplicated* pool of 50 candidates (`rrf_wide[:50]`), and apply deduplication *after* scoring. Never pre-truncate candidate pools before cross-encoding.
   - **Diversity Re-Ranking Rule**: MMR must use cosine similarity against selected document vectors and balance topical relevance ($\lambda=0.7$).

3. **Generation Pipeline (`src/generation`)**:
   - Must output structured `GenerationResult` objects containing exact `citations` (`DocumentChunk` instances).
   - All prompt templates must compel source citation via bracketed identifiers (`[Source N: doc | Page P | § Section]`).
   - Generators must implement `BaseGenerator`.

---

## 4. Ground-Truth Sanity & Assertion Rules

> [!CAUTION]
> **Never suppress the ground-truth sanity check.**

The evaluation dataset (`src/evaluation/dataset.py`) contains 14 curated queries with exact chunk index labels (`target_chunk_idx`).
- `validate_ground_truth()` asserts that each index is within bounds and contains the target document name.
- If re-chunking parameters (`max_words`, `overlap_sentences`) are altered, chunk indices **will drift**.
- If drift occurs, agents must **fail loudly** (`AssertionError`) rather than silently producing misleading zero metrics, and re-label ground truth targets using `improved-approach/label_chunks.py`.

---

## 5. Development Conventions & Constraints

1. **Windows Console Encoding**:
   All entrypoint scripts must reconfigure standard output for UTF-8 on Windows:
   ```python
   if hasattr(sys.stdout, "reconfigure"):
       sys.stdout.reconfigure(encoding="utf-8")
   ```

2. **Typing & Data Contracts**:
   - Use strongly-typed dataclasses from `src.common.types` (`DocumentChunk`, `SearchResult`, `MetricScores`, `GenerationResult`).
   - Avoid passing raw untyped tuples across pipeline boundaries.

3. **Dependencies**:
   - Sparse search: `rank-bm25` (`BM25Okapi`).
   - Dense TF-IDF: `scikit-learn` (`TfidfVectorizer(sublinear_tf=True)`).
   - Neural models: `sentence-transformers` (`all-MiniLM-L6-v2`, `cross-encoder/ms-marco-MiniLM-L-6-v2`).
   - PDF extraction: `pypdfium2` (preferred), fallback to `pdfplumber` / `pypdf`.

---

## 6. Standard Commands Reference

### Run Quantitative Benchmark (All 14 Queries × 13 Strategies)
```bash
python run_eval.py
# or:
python -m src.cli eval
```

### Interactive Multi-Strategy Document Search
```bash
python -m src.cli search "POMDP belief state filtering" --strategy rrf_dedup_mmr --top-k 5
```

### Grounded RAG Question Answering
```bash
python -m src.cli ask "How does the Binding Constraint Thesis affect harness comparisons?" --strategy rrf_dedup_mmr
```

### Corpus Ingestion
```bash
python -m src.cli ingest --corpus corpus
```

---

## 7. Current Benchmark Reference (Verified Ground Truth)

When evaluating or making changes, ensure metrics do not regress from these baseline numbers (14 queries, 2072 chunks, 11 PDFs):

| Strategy | MRR | Recall@1 | Recall@3 | Recall@5 | NDCG@5 |
|---|---|---|---|---|---|
| **1. Pure BM25 (Sparse)** | 0.573 | 0.429 | 0.714 | 0.786 | 0.612 |
| 2. Pure TF-IDF (Dense) | 0.392 | 0.143 | 0.500 | 0.714 | 0.448 |
| 3. Linear Hybrid (α=0.3) | 0.554 | 0.357 | 0.714 | 0.786 | 0.595 |
| 4. Linear Hybrid (α=0.5) | 0.524 | 0.286 | 0.714 | 0.857 | 0.599 |
| 5. Linear Hybrid (α=0.7) | 0.488 | 0.214 | 0.714 | 0.857 | 0.573 |
| 6. RRF (k=60) | 0.629 | 0.500 | 0.714 | 0.857 | 0.678 |
| 7. RRF + Deduplication | 0.629 | 0.500 | 0.714 | 0.857 | 0.678 |
| **8. RRF + Dedup + MMR** | **0.625** | **0.500** | **0.786** | **0.857** | **0.683** |
| 9. PPMI Semantic + BM25 RRF | 0.402 | 0.214 | 0.500 | 0.643 | 0.437 |
| 10. Cross-Encoder Re-rank | 0.483 | 0.286 | 0.571 | 0.857 | 0.567 |
| 11. Sentence-Transformer (MiniLM) | 0.292 | 0.143 | 0.357 | 0.571 | 0.339 |
| 12. Adaptive Hybrid | 0.494 | 0.286 | 0.571 | 0.786 | 0.549 |
| 13. SPECTER2 (Scientific Bi-Encoder) | *(pending first run)* | *(pending first run)* | *(pending first run)* | *(pending first run)* | *(pending first run)* |
