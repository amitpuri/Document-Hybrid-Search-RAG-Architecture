# Retrieval Pipeline (`src/retrieval`)

The `retrieval` package implements **Pipeline 2** of the architecture. It encapsulates all 12 hybrid search strategies across sparse keyword search, dense vector spaces, distributional semantics, neural cross-encoders, rank/score fusion, and post-processing diversity re-ranking.

---

## 📂 Module Breakdown

```
src/retrieval/
├── __init__.py          # Exports RetrievalPipeline, all retrievers, fusions, and postprocessors
├── pipeline.py          # Unified RetrievalPipeline orchestrating single-dispatch retrieval
├── retrievers/          # Core model representations
│   ├── base.py          # BaseRetriever abstract base class
│   ├── bm25.py          # BM25Okapi sparse keyword retriever (rank-bm25)
│   ├── tfidf.py         # Sklearn sublinear TF-IDF vector space model
│   ├── ppmi.py          # Distributional PPMI co-occurrence embeddings (from-scratch)
│   └── neural.py        # SentenceTransformer bi-encoder & CrossEncoder reranker
├── fusion/              # Score and rank combination algorithms
│   ├── linear.py        # Min-max normalized convex combination (alpha 0.3, 0.5, 0.7)
│   ├── rrf.py           # Reciprocal Rank Fusion (k=60)
│   └── adaptive.py      # Query-intent adaptive alpha heuristic (keyword vs natural-language)
└── postprocessing/      # Result deduplication and diversification
    ├── deduplication.py # Jaccard set-similarity sliding window deduplication
    └── mmr.py           # Maximal Marginal Relevance (MMR) diversity re-ranking
```

---

## 🎯 The 12 Retrieval Strategies

The pipeline exposes all 12 strategies evaluated in the empirical benchmark:

| # | Strategy Key | Name | Mechanism |
|---|---|---|---|
| 1 | `bm25` | **Pure BM25 (Sparse)** | Exact term frequency with length normalization ($k_1=1.5, b=0.75$). Strongest single retriever on technical jargon. |
| 2 | `tfidf` | **Pure TF-IDF (Dense)** | Sublinear TF-IDF vector space with cosine similarity. |
| 3 | `linear_0.3` | **Linear Hybrid (α=0.3)** | Min-max normalized blend: $0.3 \cdot \text{TF-IDF} + 0.7 \cdot \text{BM25}$. Favors keyword precision. |
| 4 | `linear_0.5` | **Linear Hybrid (α=0.5)** | Equal convex blend: $0.5 \cdot \text{TF-IDF} + 0.5 \cdot \text{BM25}$. |
| 5 | `linear_0.7` | **Linear Hybrid (α=0.7)** | Dense-heavy blend: $0.7 \cdot \text{TF-IDF} + 0.3 \cdot \text{BM25}$. |
| 6 | `rrf` | **RRF (k=60)** | Reciprocal Rank Fusion: $\mathrm{RRF}(d) = \sum_{m} \frac{1}{k + r_m(d) + 1}$. Immune to score scale distortions. |
| 7 | `rrf_dedup` | **RRF + Deduplication** | RRF candidate pool filtered with Jaccard similarity threshold $\ge 0.65$ to eliminate sliding-window redundancy. |
| 8 | `rrf_dedup_mmr` | **RRF + Dedup + MMR** | **Top Performer**. RRF ranking + Deduplication + Maximal Marginal Relevance ($\lambda=0.7$) balancing relevance with topic diversity. |
| 9 | `ppmi` | **PPMI Semantic + BM25 RRF** | Zero-dependency distributional semantics: term-term co-occurrence matrix + PPMI embeddings fused with BM25 via RRF. |
| 10 | `cross_encoder` | **Cross-Encoder Re-rank** | Wide pool of 50 RRF candidates re-ranked via `cross-encoder/ms-marco-MiniLM-L-6-v2`, followed by Jaccard deduplication. |
| 11 | `sentence_transformer` | **Sentence-Transformer (MiniLM)** | Pure dense retrieval using `all-MiniLM-L6-v2` bi-encoder embeddings and cosine similarity. |
| 12 | `adaptive` | **Adaptive Hybrid** | Query-intent heuristic: chooses $\alpha=0.3$ for keyword/jargon queries ($\le 4$ tokens) and $\alpha=0.7$ for natural language questions. |

---

## 🧮 Mathematical Formulations

### 1. Reciprocal Rank Fusion (RRF)
$$\mathrm{RRF}(d) = \sum_{m \in \{\mathrm{BM25}, \mathrm{Dense}\}} \frac{1}{k + r_m(d) + 1} \quad (k=60)$$

### 2. Maximal Marginal Relevance (MMR)
$$\mathrm{MMR}(d) = \arg\max_{d \in R \setminus S} \left[ \lambda \cdot \mathrm{Sim}(d, q) - (1 - \lambda) \max_{s \in S} \mathrm{Sim}(d, s) \right]$$
- $\lambda = 0.7$ balances high topical relevance with penalizing near-identical passages.

### 3. Positive Pointwise Mutual Information (PPMI)
$$\mathrm{PPMI}(w_1, w_2) = \max\left(0, \log_2 \frac{P(w_1, w_2)}{P(w_1) P(w_2)}\right)$$
Distributional word embeddings are formed by keeping the top-50 strongest context dimensions, and chunk vectors are mean-pooled.

---

## 💡 Usage Example

```python
from src.ingestion import IngestionPipeline
from src.retrieval import RetrievalPipeline

# Ingest corpus
chunk_store, _, _ = IngestionPipeline().run()

# Initialize and index retrieval pipeline
pipeline = RetrievalPipeline(chunk_store)
pipeline.index()

# Search using top-performing strategy
results = pipeline.search(
    query="Dynamic Tiered AgentRunner Framework Risk Adaptive Tiering",
    strategy="rrf_dedup_mmr",
    top_k=5
)

for r in results:
    print(f"Rank #{r.rank} [{r.strategy}] -> {r.chunk.doc_name} (Page {r.chunk.page_num})")
    print(f"Text snippet: {r.chunk.text[:120]}...\n")
```
