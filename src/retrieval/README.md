# Retrieval Pipeline (`src/retrieval`)

The `retrieval` package implements **Pipeline 2** of the architecture. It encapsulates all **15 + 3 = 18 hybrid search strategies** (15 core strategies + 3 P0 ablation cells) across sparse keyword search, dense vector spaces, distributional semantics, domain-adapted scientific bi-encoders, neural cross-encoders, knowledge-graph traversal, rank/score fusion, post-processing diversity re-ranking, and **P0 ablation cells** for factorial isolation of Graph, Dedup, and MMR contributions.

---

## 📂 Module Breakdown

```
src/retrieval/
├── __init__.py          # Exports RetrievalPipeline, all retrievers, fusions, and postprocessors
├── pipeline.py          # Unified RetrievalPipeline orchestrating single-dispatch retrieval across 17 strategies (14 original + 3 P0 ablation cells)
├── retrievers/          # Core model representations
│   ├── base.py          # BaseRetriever abstract base class
│   ├── bm25.py          # BM25Okapi sparse keyword retriever (rank-bm25)
│   ├── tfidf.py         # Sklearn sublinear TF-IDF vector space model
│   ├── ppmi.py          # Distributional PPMI co-occurrence embeddings (from-scratch)
│   ├── neural.py        # MiniLM & SPECTER2 bi-encoders and CrossEncoder reranker
│   ├── qdrant.py        # Qdrant vector database retriever (approximate nearest neighbor HNSW)
│   └── graph.py         # Knowledge Graph retriever (1-hop traversal & community detection)
├── fusion/              # Score and rank combination algorithms
│   ├── linear.py        # Min-max normalized convex combination (alpha 0.3, 0.5, 0.7)
│   ├── rrf.py           # Reciprocal Rank Fusion (k=60, supports 2-way and 3-way fusion)
│   └── adaptive.py      # Query-intent adaptive alpha heuristic (keyword vs natural-language)
└── postprocessing/      # Result deduplication and diversification
    ├── deduplication.py # Jaccard set-similarity sliding-window deduplication
    └── mmr.py           # Maximal Marginal Relevance (MMR) diversity re-ranking
```

---

## 🎯 The Retrieval Strategies

The pipeline exposes all retrieval strategies evaluated in the repository:

| # | Strategy Key | Name | Mechanism |
|---|---|---|---|
| 1 | `bm25` | **Pure BM25 (Sparse)** | Exact term frequency with length normalization ($k_1=1.5, b=0.75$). Strongest baseline on technical domain jargon. |
| 2 | `tfidf` | **Pure TF-IDF (Sparse Vector Space)** | Sublinear TF-IDF vector space with cosine similarity. Sparse, term-based representation. |
| 3 | `linear_0.3` | **Linear Hybrid (α=0.3)** | Min-max normalized blend: $0.3 \cdot \text{TF-IDF} + 0.7 \cdot \text{BM25}$. Favors keyword precision. |
| 4 | `linear_0.5` | **Linear Hybrid (α=0.5)** | Equal convex blend: $0.5 \cdot \text{TF-IDF} + 0.5 \cdot \text{BM25}$. |
| 5 | `linear_0.7` | **Linear Hybrid (α=0.7)** | Dense-heavy blend: $0.7 \cdot \text{TF-IDF} + 0.3 \cdot \text{BM25}$. |
| 6 | `rrf` | **RRF (k=60)** | Reciprocal Rank Fusion: $\mathrm{RRF}(d) = \sum_{m} \frac{1}{k + r_m(d) + 1}$. Immune to score scale distortions. |
| 7 | `rrf_dedup` | **RRF + Deduplication** | RRF candidate pool filtered with Jaccard similarity threshold $\ge 0.65$ to eliminate sliding-window redundancy. |
| 8 | `rrf_dedup_mmr` | **RRF + Dedup + MMR** | RRF ranking + Deduplication + Maximal Marginal Relevance ($\lambda=0.7$) balancing relevance with topic diversity. Top non-graph performer (**0.683 NDCG@5**). |
| 9 | `ppmi` | **PPMI Semantic + BM25 RRF** | Zero-dependency distributional semantics: term-term co-occurrence matrix + PPMI embeddings fused with BM25 via RRF. |
| 10 | `cross_encoder` | **Cross-Encoder Re-rank** | Wide pool of 50 un-deduplicated candidates re-ranked via `cross-encoder/ms-marco-MiniLM-L-6-v2`, followed by Jaccard deduplication. |
| 11 | `sentence_transformer` | **Sentence-Transformer (MiniLM)** | Pure dense retrieval using `all-MiniLM-L6-v2` bi-encoder embeddings and cosine similarity. |
| 12 | `adaptive` | **Adaptive Hybrid** | Query-intent heuristic: chooses $\alpha=0.3$ for keyword/jargon queries ($\le 4$ tokens) and $\alpha=0.7$ for natural language questions. |
| 13 | `specter2` | **SPECTER2 (Scientific Bi-Encoder)** | AllenAI domain-adapted scientific embeddings via `allenai/specter2_base` with `allenai/specter2_proximity` adapter. |
| 14 | `rrf_graph_dedup_mmr` | **RRF + Graph + Dedup + MMR** ★ KG | **Repository Best**. Tri-fusion combining BM25, TF-IDF, and 1-hop NetworkX Knowledge Graph traversal via calibrated RRF, followed by sliding-window Jaccard deduplication and MMR re-ranking (**0.667 MRR**, **0.571 Recall@1**, **0.714 NDCG@5**, **0.643 Relation Coverage**). |
| 15 | `qdrant` / `qdrant_vector` | **Qdrant Vector (ANN)** | High-speed approximate nearest neighbor (ANN) vector retrieval powered by local or remote Qdrant HNSW vector index. Sub-millisecond similarity scoring without holding large dense matrices in Python RAM. |
|| 16 | `graph_only` | **Graph Only (P0 Ablation)** | Graph retrieval alone, no BM25/TF-IDF, no postprocessing. Isolates whether the Knowledge Graph carries standalone retrieval signal. **✅ COMPLETED 2026-09-18** |
|| 17 | `rrf_graph` | **RRF + Graph (P0 Ablation)** | RRF fusion of BM25 + TF-IDF + Graph, no dedup/MMR. Isolates the graph's raw contribution to fusion, pre-dedup/MMR. **✅ COMPLETED 2026-09-18** |
|| 18 | `rrf_graph_dedup` | **RRF + Graph + Dedup (P0 Ablation)** | RRF + Graph + Jaccard dedup, no MMR. Isolates MMR's specific marginal contribution on top of graph fusion. **✅ COMPLETED 2026-09-18** |

---

## 🧮 Mathematical Formulations

### 1. Reciprocal Rank Fusion (RRF)
$$\mathrm{RRF}(d) = \sum_{m \in M} \frac{1}{k + r_m(d) + 1} \quad (k=60)$$
Where $M$ is the set of active ranking signals (e.g., $\{\mathrm{BM25}, \mathrm{TF\text{-}IDF}\}$ for Strategy 6–8, or $\{\mathrm{BM25}, \mathrm{TF\text{-}IDF}, \mathrm{Graph}\}$ for Strategy 14).

### 2. Maximal Marginal Relevance (MMR)
$$\mathrm{MMR}(d) = \arg\max_{d \in R \setminus S} \left[ \lambda \cdot \mathrm{Sim}(d, q) - (1 - \lambda) \max_{s \in S} \mathrm{Sim}(d, s) \right]$$
- $\lambda = 0.7$ balances high topical relevance with penalizing near-identical passages.

### 3. Knowledge Graph 1-Hop Activation & Traversal
$$\mathrm{Score}_{\mathrm{KG}}(c) = \sum_{e \in E(q) \cap E(c)} \mathrm{IDF}(e) \cdot \log(1 + \deg(e)) + \sum_{e' \in N_1(E(q)) \cap E(c)} \frac{\mathrm{IDF}(e')}{1 + \mathrm{dist}(e, e')}$$
- Direct entity matches are weighted by corpus-level inverse document frequency ($\mathrm{IDF}$) and node degree centrality.
- 1-hop neighbor concepts ($N_1$) propagate activation with distance decay, surfacing structurally related passages even when lexical tokens do not overlap directly.
- Disconnected queries fall back to Louvain community detection cluster activations.

### 4. Positive Pointwise Mutual Information (PPMI)
$$\mathrm{PPMI}(w_1, w_2) = \max\left(0, \log_2 \frac{P(w_1, w_2)}{P(w_1) P(w_2)}\right)$$
Distributional word embeddings are formed by retaining the top-50 strongest context dimensions, with passage vectors computed via mean pooling.

---

## 💡 Usage Example

```python
from src.ingestion import IngestionPipeline
from src.retrieval import RetrievalPipeline

# Ingest corpus and build indexes (Parquet store + Knowledge Graph)
chunk_store, _, _ = IngestionPipeline().run()

# Initialize and index retrieval pipeline across all strategies (including Qdrant ANN)
pipeline = RetrievalPipeline(chunk_store)
pipeline.index()

# 1. Search using Strategy 14 (Repository Best: RRF + Graph + Dedup + MMR)
results_kg = pipeline.search(
    query="Dynamic Tiered AgentRunner Framework Risk Adaptive Tiering",
    strategy="rrf_graph_dedup_mmr",
    top_k=5
)

for r in results_kg:
    print(f"Rank #{r.rank} [{r.strategy}] -> {r.chunk.doc_name} (Page {r.chunk.page_num}, § {r.chunk.section})")
    print(f"Snippet: {r.chunk.text[:120]}...\n")

# 2. Search using Strategy 8 (RRF + Dedup + MMR)
results_mmr = pipeline.search(
    query="How does the Binding Constraint Thesis affect harness comparisons?",
    strategy="rrf_dedup_mmr",
    top_k=5
)

# 3. Search using Strategy 15 (Qdrant Vector ANN)
results_qdrant = pipeline.search(
    query="POMDP belief state filtering",
    strategy="qdrant",
    top_k=5
)
```
