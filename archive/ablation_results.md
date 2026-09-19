| Strategy | MRR | Recall@1 | Recall@3 | Recall@5 | NDCG@5 | EntCov | Status |
|---|---|---|---|---|---|---|---|
| `bm25` -- Pure BM25 (Sparse) | 0.899 | 0.857 | 0.929 | 1.000 | 0.924 | 0.857 | ran |
| `tfidf` -- Pure TF-IDF (Dense) | 0.893 | 0.857 | 0.929 | 0.929 | 0.902 | 0.845 | ran |
| `rrf` -- RRF: BM25 + TF-IDF | 0.911 | 0.857 | 0.929 | 1.000 | 0.933 | 0.869 | ran |
| `rrf_dedup` -- RRF + Dedup | 0.911 | 0.857 | 0.929 | 1.000 | 0.933 | 0.869 | ran |
| `rrf_dedup_mmr` -- RRF + Dedup + MMR (Strategy 8) | 0.917 | 0.857 | 1.000 | 1.000 | 0.938 | 0.893 | ran |
| `graph_only` | - | - | - | - | - | - | not yet registered |
| `rrf_graph` | - | - | - | - | - | - | not yet registered |
| `rrf_graph_dedup` | - | - | - | - | - | - | not yet registered |
| `rrf_graph_dedup_mmr` -- RRF + Dedup + MMR + Graph (Strategy 14) | 0.964 | 0.929 | 1.000 | 1.000 | 0.974 | 0.893 | ran |
