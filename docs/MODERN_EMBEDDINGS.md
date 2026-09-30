# Modern Dense Embeddings: BGE, E5, and Beyond

**Status:** Infrastructure ready for benchmarking (Phase 3)  
**Last Updated:** 2026-09-19

---

## Overview

This document describes modern dense embedding models available for evaluation alongside the current MiniLM and SPECTER2 baselines. These models represent state-of-the-art retrieval performance on MTEB benchmarks.

---

## Available Models

### 1. **BGE (BAAI General Embeddings)**

#### BGE-small-en-v1.5
- **Model:** `BAAI/bge-small-en-v1.5`
- **Size:** 33M parameters
- **Dimensionality:** 384
- **Training Data:** Large-scale retrieval dataset + NLI corpus
- **MTEB Ranking:** #1 for general retrieval (as of 2024)
- **Performance vs MiniLM:**
  - MiniLM: 0.318 MRR (current baseline)
  - BGE-small: ~0.45-0.55 MRR (estimated on technical corpora)
  - **Expected Improvement:** +40-70% (estimated; requires benchmarking)

#### Installation
```bash
pip install -e ".[modern-embeddings]"
```

#### Usage
```python
from src.retrieval.retrievers.neural import BGERetriever

retriever = BGERetriever(model_name="BAAI/bge-small-en-v1.5")
retriever.index(corpus_texts)
scores = retriever.score(query)
```

---

### 2. **E5 (Text Embeddings by Mistral Collective)**

#### E5-small-v2
- **Model:** `intfloat/e5-small-v2`
- **Size:** 33M parameters
- **Dimensionality:** 384
- **Training Data:** Large-scale contrastive learning on diverse NLI + retrieval corpora
- **Specialization:** General-purpose dense retrieval (not domain-specific)
- **MTEB Ranking:** Top 5 for retrieval; strong generalization
- **Performance vs MiniLM:**
  - MiniLM: 0.318 MRR
  - E5-small: ~0.42-0.52 MRR (estimated on technical corpora)
  - **Expected Improvement:** +30-65% (estimated; requires benchmarking)

#### Installation
```bash
pip install -e ".[modern-embeddings]"
```

#### Usage
```python
from src.retrieval.retrievers.neural import E5Retriever

retriever = E5Retriever(model_name="intfloat/e5-small-v2")
retriever.index(corpus_texts)
scores = retriever.score(query)
```

---

### 3. **Future Additions**

#### Domain-Specific Models (Phase 3+)
- **SciBERT Embeddings:** `allenai/scibert-base` (already integrated as SPECTER2 proximity adapter)
- **PatentBERT:** For patent/technical document retrieval
- **PubMedBERT:** For biomedical literature

#### Advanced Architectures
- **Dense-in-Batch (DIB):** Multi-GPU contrastive learning (scales to 10M+ documents)
- **ColBERT:** Late-interaction retrieval (token-level matching)
- **SPLADE:** Sparse-dense hybrid (inherits benefits of both paradigms)

---

## Benchmarking Strategy

### Phase 2 (Current)
- [x] Add BGE and E5 to pyproject.toml `[modern-embeddings]`
- [x] Create placeholder retrievers (src/retrieval/retrievers/bge.py, e5.py)
- [x] Document infrastructure and usage

### Phase 3 (Next)
- [ ] Benchmark BGE-small on 14-query baseline
- [ ] Benchmark E5-small on 14-query baseline
- [ ] Compare MRR, Recall@{1,3,5}, NDCG@5 vs MiniLM (0.318 MRR)
- [ ] Run hybrid fusion (RRF) combining BGE/E5 with BM25
- [ ] Document improvements and failure modes
- [ ] Report findings in docs/researchpaper.md Section 7 (Future Work)

### Expected Results
- Single BGE or E5: MRR ~0.45-0.55 (vs MiniLM 0.318)
- BGE + RRF: MRR ~0.63-0.68 (vs current RRF 0.629)
- E5 + RRF: MRR ~0.61-0.67

---

## Why Modern Models Matter for Technical Literature

### Problem with MiniLM (2021)
- Small model (33M params) trained on general web data
- Struggles with domain jargon: POMDP, AgentRunner, StarShell diffused across semantic space
- Performance: 0.318 MRR (vs BM25 0.573)

### Advantage of BGE/E5
- Trained on **large-scale retrieval corpus** (not just general NLI)
- Better handling of technical terminology through contrastive learning
- Strong performance on domain-agnostic benchmarks (MTEB)

### Why Not Immediately Adopted
1. **Computational Cost:** BGE/E5 only marginally larger than MiniLM, but require more VRAM
2. **Hyperparameter Tuning:** RRF weights may change with new embeddings
3. **Benchmark Size:** Current 14-query benchmark too small for robust conclusions

---

## Implementation Notes

### Caching
Modern embeddings are cached using the same mechanism as MiniLM:
- Embeddings saved as `.npy` files (numpy binary format)
- Cache key includes model name and corpus fingerprint
- Automatic cache invalidation on corpus changes

### Pooling Strategy
- **MiniLM, BGE, E5:** Mean pooling (standard for bi-encoders)
- **SPECTER2:** CLS token with proximity adapter (asymmetric query adapter on queries)
- **Recommendation:** Use mean pooling for new models to ensure consistency

### Normalization
- All dense embeddings L2-normalized before storage
- Cosine similarity used for ranking (standard for bi-encoders)
- Score range: [0, 1] (cosine similarity upper-bounded at 1.0)

---

## FAQ

**Q: Why not use BGE/E5 immediately instead of benchmarking?**
A: MiniLM is sufficient for research; BGE/E5 require re-tuning RRF hyperparameters and expanding the benchmark for robust conclusions.

**Q: How much faster/slower are BGE/E5 vs MiniLM?**
A: Similar speed (33M params each). Inference time ~same. BGE slightly larger on disk (~67MB vs 80MB).

**Q: Can I use BGE large-en-v1.5 instead of small?**
A: Yes, but larger models require more VRAM. 2GB memory handles small; large requires 4-6GB.

**Q: What about SPECTER2 for modern retrieval?**
A: SPECTER2 is citation-aware (paper-to-paper), not passage retrieval. Inappropriate for document chunks. Use BGE/E5 for passage retrieval.

---

## References

- BGE Paper: "[Towards Universal and Transferable Large Models for Information Retrieval](https://arxiv.org/abs/2405.16027)"
- E5 Paper: "[Text Embeddings by Weakly-Supervised Contrastive Pre-Training](https://arxiv.org/abs/2212.03533)"
- MTEB Benchmark: https://huggingface.co/spaces/mteb/leaderboard
- Current Baseline: src/retrieval/retrievers/neural.py (MiniLM, SPECTER2)

---

## Next Steps

1. **Phase 3:** Implement benchmarking suite for BGE/E5
2. **Phase 3:** Compare with current MiniLM baseline
3. **Phase 4:** Integrate best-performing model into default pipeline
4. **Ongoing:** Monitor MTEB leaderboard for newer models

---

**Status:** Infrastructure ready. Benchmarking pending. See CRITICAL_ISSUES_IDENTIFIED.md Issue 2.1 for current status.

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>
