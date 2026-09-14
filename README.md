# Document Hybrid Search & RAG Architecture

A modular, enterprise-grade hybrid document search and retrieval-augmented generation (RAG) platform built over scientific and technical literature.

---

## 🏗️ Three-Pipeline Architecture (`src/`)

The system is segregated into three decoupled pipelines:

```
src/
├── ingestion/                   # PIPELINE 1: INGESTION (Big-Data / Batch Ready)
│   ├── extractors.py            # Multi-backend PDF extraction (pypdfium2, pdfplumber, pypdf)
│   ├── chunkers.py              # Sentence-aware & active section-header preserving chunking
│   ├── cache.py                 # SHA-256 parameter- & mtime-sensitive cryptographic disk cache
│   ├── storage.py               # Decoupled chunk storage abstractions (InMemory / Big-Data ready)
│   └── pipeline.py              # IngestionPipeline orchestrator
│
├── retrieval/                   # PIPELINE 2: RETRIEVAL (12 Benchmark Strategies)
│   ├── retrievers/              # BM25, TF-IDF, PPMI Distributional Semantics, MiniLM Bi-Encoder, Cross-Encoder
│   ├── fusion/                  # Convex Linear Combination, Reciprocal Rank Fusion (RRF), Adaptive Hybrid
│   ├── postprocessing/          # Jaccard Result Deduplication, Maximal Marginal Relevance (MMR)
│   └── pipeline.py              # RetrievalPipeline orchestrator & single dispatch
│
├── generation/                  # PIPELINE 3: GENERATION (Future / RAG Ready)
│   ├── context.py               # ContextBuilder with bracketed source provenance headers
│   ├── prompts.py               # Grounded instruction prompt templates
│   ├── base.py                  # BaseGenerator abstract class
│   ├── mock.py                  # GroundedSynthesisGenerator (local offline citations synthesizer)
│   └── pipeline.py              # GenerationPipeline orchestrator
│
├── evaluation/                  # Quantitative IR benchmarking suite
│   ├── metrics.py               # MRR, Recall@1/3/5, NDCG@5
│   ├── dataset.py               # 14 curated benchmark queries & ground-truth validation
│   └── harness.py               # Benchmark runner across all 12 strategies
│
├── engine.py                    # Unified HybridSearchEngine binding all 3 pipelines
└── cli.py                       # Unified CLI (eval, search, ask, ingest)
```

---

## 🚀 Quickstart

### Installation

```bash
pip install -r requirements.txt
```

### 1. Run Quantitative Benchmark (14 queries × 12 strategies)

```bash
python run_eval.py
# or:
python -m src.cli eval
```

### 2. Interactive Search CLI

```bash
python -m src.cli search "POMDP belief state filtering" --strategy rrf_dedup_mmr --top-k 5
```

Supported strategy flags:
- `bm25` (Pure BM25 Sparse)
- `tfidf` (Pure TF-IDF Dense)
- `linear_0.3`, `linear_0.5`, `linear_0.7` (Linear Hybrid)
- `rrf` (Reciprocal Rank Fusion)
- `rrf_dedup` (RRF with Jaccard deduplication)
- `rrf_dedup_mmr` (RRF + Dedup + MMR diversity re-ranking — **Top Performer**)
- `ppmi` (Distributional PPMI Semantic + BM25 RRF)
- `cross_encoder` (Cross-Encoder Re-rank over wide candidate pool)
- `sentence_transformer` (Sentence-Transformer MiniLM Dense)
- `adaptive` (Adaptive Hybrid Heuristic)

### 3. Grounded Question Answering (RAG Pipeline)

```bash
python -m src.cli ask "What risk-tiering mechanisms does the AgentRunner framework apply?" --strategy rrf_dedup_mmr
```

### 4. Corpus Ingestion

```bash
python -m src.cli ingest --corpus corpus
```

---

## 📊 Empirical Benchmark Results

Measured across 14 ground-truth queries on 11 research PDFs (354 pages, 2,072 structured chunks):

| Retrieval Strategy | MRR | Recall@1 | Recall@3 | Recall@5 | NDCG@5 |
|---|---|---|---|---|---|
| **1. Pure BM25 (Sparse)** | 0.573 | 0.429 | 0.714 | 0.786 | 0.612 |
| 2. Pure TF-IDF (Dense) | 0.392 | 0.143 | 0.500 | 0.714 | 0.448 |
| 3. Linear Hybrid (α=0.3) | 0.554 | 0.357 | 0.714 | 0.786 | 0.595 |
| 4. Linear Hybrid (α=0.5) | 0.524 | 0.286 | 0.714 | 0.857 | 0.599 |
| 5. Linear Hybrid (α=0.7) | 0.488 | 0.214 | 0.714 | 0.857 | 0.573 |
| 6. RRF (k=60) | 0.629 | **0.500** | 0.714 | 0.857 | 0.678 |
| 7. RRF + Deduplication | 0.629 | **0.500** | 0.714 | 0.857 | 0.678 |
| **8. RRF + Dedup + MMR** | **0.625** | **0.500** | **0.786** | 0.857 | **0.683** |
| 9. PPMI Semantic + BM25 RRF | 0.402 | 0.214 | 0.500 | 0.643 | 0.437 |
| 12. Adaptive Hybrid | 0.494 | 0.286 | 0.571 | 0.786 | 0.549 |
| 10. Cross-Encoder Re-rank | 0.483 | 0.286 | 0.571 | 0.857 | 0.567 |
| 11. Sentence-Transformer (MiniLM) | 0.292 | 0.143 | 0.357 | 0.571 | 0.339 |

---

## 🐍 Python API Usage

```python
from src.engine import HybridSearchEngine

# 1. Initialize engine (loads cache or ingests corpus)
engine = HybridSearchEngine.from_corpus("corpus/")

# 2. Execute Hybrid Search
results = engine.search("POMDP belief state filtering", strategy="rrf_dedup_mmr", top_k=3)
for res in results:
    print(f"Rank {res.rank}: {res.chunk.doc_name} (Page {res.chunk.page_num})")
    print(f"Excerpt: {res.chunk.text[:120]}...\n")

# 3. Grounded RAG Generation
response = engine.generate_answer(
    "What risk-tiering mechanisms does the AgentRunner framework apply?",
    strategy="rrf_dedup_mmr"
)
print(response.answer)
```
