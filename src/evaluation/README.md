# Evaluation Suite & Benchmarking (`src/evaluation`)

The `evaluation` package provides quantitative, empirically validated information retrieval (IR) benchmarking over the document corpus. It tests all 13 hybrid retrieval strategies against curated ground-truth targets.

---

## 📂 Module Breakdown

```
src/evaluation/
├── __init__.py      # Exports EvaluationHarness, evaluate_ranking, EVAL_DATASET, validate_ground_truth
├── metrics.py       # Mathematical implementations of MRR, Recall@K, and NDCG@5
├── dataset.py       # 14 curated benchmark queries with chunk-level ground truth targets
└── harness.py       # EvaluationHarness orchestrator running queries across all 13 strategies
```

---

## 📊 Evaluation Metrics (`metrics.py`)

### 1. Mean Reciprocal Rank (MRR)
$$\mathrm{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{r_i}$$
Where $r_i$ is the 1-indexed position of the *first relevant chunk* found for query $i$. If no relevant passage is retrieved, the reciprocal rank is $0$.

### 2. Recall@K (K = 1, 3, 5)
$$\mathrm{Recall}@K = \begin{cases} 1.0 & \text{if a relevant chunk appears within the top } K \\ 0.0 & \text{otherwise} \end{cases}$$

### 3. Normalized Discounted Cumulative Gain (NDCG@5)
$$\mathrm{DCG}_5 = \sum_{r=1}^{5} \frac{\mathrm{rel}(r)}{\log_2(r + 1)}, \quad \mathrm{NDCG}_5 = \frac{\mathrm{DCG}_5}{\mathrm{IDCG}_5}$$
Where $\mathrm{IDCG}_5 = \frac{1.0}{\log_2(1 + 1)} = 1.0$ (representing the ideal ranking with the relevant chunk at rank 1).

### 4. Fine-Grained Chunk-Level Relevance (`is_relevant`)
Prior coarse evaluations credited accidental document hits on bibliographies or author lists. This suite checks exact chunk indices (`target_chunk_idx`), ensuring metrics reflect true passage discovery.

---

## 🎯 Benchmark Dataset (`dataset.py`)

Comprises **14 curated queries** targeting specific concepts across 11 PDFs:
- **Queries 1–10 (Keyword Phrased)**: Target specific theorems, frameworks, and mechanisms (e.g., *"Binding Constraint Thesis"*, *"StarShell"*, *"POMDP belief state"*).
- **Queries 11–14 (Natural Language Phrased)**: Question-formatted queries (*"How does…", "What does…"*) targeting the same validated ground-truth chunks to evaluate query-intent adaptive alpha selection.
- **`validate_ground_truth()` Sanity Assertion**: Fails loudly on startup if any target chunk index drifts or points to the wrong document, preventing silent evaluation decay.

---

## 📈 Empirical Results (14 Queries × 13 Strategies)

Measured on 11 PDFs (354 pages, 2,072 structured chunks):

| Retrieval Strategy | MRR | Recall@1 | Recall@3 | Recall@5 | NDCG@5 | Key Insight |
|---|---|---|---|---|---|---|
| **1. Pure BM25 (Sparse)** | 0.573 | 0.429 | 0.714 | 0.786 | 0.612 | Exceptional on precise technical acronyms and named entities. |
| 2. Pure TF-IDF (Dense) | 0.392 | 0.143 | 0.500 | 0.714 | 0.448 | Outperformed by BM25 due to lack of saturation limits. |
| 3. Linear Hybrid (α=0.3) | 0.554 | 0.357 | 0.714 | 0.786 | 0.595 | Best linear weighting; strongly favors sparse keyword signal. |
| 4. Linear Hybrid (α=0.5) | 0.524 | 0.286 | 0.714 | 0.857 | 0.599 | Equal convex combination. |
| 5. Linear Hybrid (α=0.7) | 0.488 | 0.214 | 0.714 | 0.857 | 0.573 | Dense-heavy combination; degraded by dense score noise. |
| 6. RRF (k=60) | 0.629 | **0.500** | 0.714 | 0.857 | 0.678 | Dominates linear fusion; rank reciprocals prevent outlier score skews. |
| 7. RRF + Deduplication | 0.629 | **0.500** | 0.714 | 0.857 | 0.678 | Removes overlapping sliding-window fragments without metric loss. |
| **8. RRF + Dedup + MMR** | **0.625** | **0.500** | **0.786** | **0.857** | **0.683** | **Overall Top Performer**. Achieves highest NDCG@5 and Recall@3 through diversity re-ranking. |
| 9. PPMI Semantic + BM25 RRF | 0.402 | 0.214 | 0.500 | 0.643 | 0.437 | From-scratch distributional semantics without neural weights. |
| 10. Cross-Encoder Re-rank | 0.483 | 0.286 | 0.571 | 0.857 | 0.567 | Re-ranks 50 un-deduplicated candidates. Represents out-of-domain ceiling. |
| 11. Sentence-Transformer (MiniLM) | 0.292 | 0.143 | 0.357 | 0.571 | 0.339 | Generic bi-encoders diffuse niche jargon (*"StarShell"*, *"POMDP"*). |
| 12. Adaptive Hybrid | 0.494 | 0.286 | 0.571 | 0.786 | 0.549 | Heuristic fires correctly; lands predictably between α=0.3 and α=0.7. |
| 13. SPECTER2 (Scientific Bi-Encoder) | *(pending first run)* | *(pending first run)* | *(pending first run)* | *(pending first run)* | *(pending first run)* | Domain-adapted scientific embeddings via `allenai/specter2_proximity`. |

---

## 🚀 Running the Evaluation

### Via CLI:
```bash
python run_eval.py
# or:
python -m src.cli eval
```

### Via Python API:
```python
from src.evaluation import EvaluationHarness

harness = EvaluationHarness()
summary_metrics = harness.run()

print(f"Top Strategy MRR: {summary_metrics['8. RRF + Dedup + MMR']['mrr']:.3f}")
```
