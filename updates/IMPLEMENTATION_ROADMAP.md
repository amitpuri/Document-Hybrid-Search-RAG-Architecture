# Implementation Roadmap - Critical Issues

**Status:** PLANNING  
**Priority:** HIGH - Addresses reproducibility, statistical rigor, and production readiness  
**Owner:** Development team  
**Last Updated:** 2026-09-18

---

## Executive Summary

This roadmap addresses 15 critical issues identified in code review feedback, organized by implementation phase. Focus is on rebuilding statistical rigor (expanding benchmark from N=22 to N=100+), fixing undeclared pickle security issues, clarifying documentation, and refactoring dependencies.

---

## Phase 1: Immediate Documentation Fixes (1-2 days)

### P1.1: Fix Contradictory Claims ✅ DONE

**Status:** COMPLETED  
**Changes:**
- [x] Updated AGENTS.md: Removed "consistently outperforms" claim, added caveats about dense model underperformance
- [x] Updated README selection guide: Added "Evidence-Based" label, documented that hybrid fusion does NOT consistently beat BM25
- [x] Fixed TF-IDF label: Renamed "Pure TF-IDF (Dense)" → "Pure TF-IDF (Sparse)"
- [x] Added WARNING block: Statistical rigor caveat with 0.045 MRR threshold
- [x] Documented implicit hyperparameter tuning on test set

**Evidence:** With N=22 queries, BM25 (0.551 MRR) beats linear hybrids (0.471–0.503), RRF degrades after expansion, cross-encoder specializes in multi-hop.

---

### P1.2: Inspect Reranker Performance

**Status:** IN PROGRESS  
**Task:** Debug why Cross-Encoder MRR (0.481) < Recall@5 of pool (0.857)

**Steps:**
1. [ ] Verify score normalization in `src/retrieval/retrievers/neural.py`:
   - [ ] Check if BM25 scores (0-20+ unbounded) normalized vs cosine (0-1 bounded)?
   - [ ] Is min-max normalization applied before cross-encoder reranking?
   - [ ] What's the truncation point (50 candidates)?

2. [ ] Manual inspection: Run 5 queries where reranker drops performance
   ```bash
   python -m src.cli search "query" --strategy bm25 --top-k 5
   python -m src.cli search "query" --strategy cross_encoder --top-k 5
   # Compare: does target chunk move down in reranking?
   ```

3. [ ] Check domain mismatch: ms-marco model trained on web, applied to technical PDFs

4. [ ] Document findings in `src/retrieval/retrievers/neural.py` as TODO comment

---

## Phase 2: Dependency & Security Fixes (3-5 days)

### P2.1: Replace Unsafe Pickle Cache with Parquet

**Status:** BLOCKED (needs Phase 2.2 first)  
**Severity:** HIGH (security)

**Problem:** Cache uses pickle which is unsafe to load from untrusted/shared directories

**Files to modify:**
- `src/ingestion/cache.py` (lines 140-180: pickle.load/dump)
- `src/retrieval/retrievers/ppmi.py` (lines 150-160: pickle state)

**Implementation:**
1. [ ] Create new cache functions: `load_cache_parquet()` and `save_cache_parquet()`
2. [ ] Keep backward compatibility: Try to load .pkl, fall back to parquet
3. [ ] Add migration script: `scripts/migrate_pkl_to_parquet.py`
4. [ ] Update docs: "Legacy .pkl caches are deprecated; use `scripts/migrate_pkl_to_parquet.py`"

**Testing:**
```bash
# Verify new parquet cache works
python -m src.cli ingest --corpus corpus --force

# Verify migration script
python scripts/migrate_pkl_to_parquet.py
```

---

### P2.2: Refactor Dependencies into pyproject.toml

**Status:** READY  
**Task:** Move from requirements.txt + hard deps → pyproject.toml with extras

**Current state:**
- All LLM SDKs (openai, anthropic, google-genai, boto3, google-cloud-aiplatform) hardcoded
- torch and sentence-transformers hardcoded (very large)
- pdfplumber and pypdf commented out (but promised as fallbacks)

**Target state:**
```toml
[project]
dependencies = [
    "pypdfium2>=4.0.0,<5.0.0",
    "numpy>=1.24.0,<2.0.0",
    "scipy>=1.10.0,<2.0.0",
    "scikit-learn>=1.2.0,<2.0.0",
    "rank-bm25>=0.2.2,<1.0.0",
    "networkx>=3.0,<4.0",
    "qdrant-client>=1.11.0,<2.0.0",
    "sentence-transformers>=2.2.0,<3.0.0",
    "torch>=2.0.0,<3.0.0",
    "transformers>=4.35.0,<5.0.0",
    "python-dotenv>=1.0.0,<2.0.0",
]

[project.optional-dependencies]
retrieval = [
    # same as above
]
llm-anthropic = ["anthropic>=0.34.0,<1.0.0"]
llm-openai = ["openai>=1.0.0,<2.0.0"]
llm-gemini = ["google-genai>=1.0.0,<2.0.0"]
bedrock = ["boto3>=1.28.0,<2.0.0"]
vertex = ["google-cloud-aiplatform>=1.38.0,<2.0.0"]
all = [
    "anthropic>=0.34.0,<1.0.0",
    "openai>=1.0.0,<2.0.0",
    "google-genai>=1.0.0,<2.0.0",
    "boto3>=1.28.0,<2.0.0",
    "google-cloud-aiplatform>=1.38.0,<2.0.0",
]
fallback-pdf = [
    "pdfplumber>=0.10.0,<1.0.0",
    "pypdf>=3.0.0,<4.0.0",
]
```

**Implementation:**
1. [ ] Create `pyproject.toml` with above structure
2. [ ] Keep `requirements.txt` for backward compatibility (legacy workflows)
3. [ ] Add `requirements-lock.txt` for reproducible installs
4. [ ] Update README: Add "Installation" section with extras examples
5. [ ] Update AGENTS.md: Document which components are optional

**Testing:**
```bash
# Core retrieval only
pip install -e ".[retrieval]"
python -m src.cli search "test query" --strategy bm25

# With Anthropic
pip install -e ".[retrieval,llm-anthropic]"
python -m src.cli ask "test question" --provider anthropic

# All providers
pip install -e ".[all]"

# Verify fallback PDF readers work
pip install -e ".[fallback-pdf]"
```

---

### P2.3: Document Generation Status (NOT Future, Already Production)

**Status:** READY  
**Issue:** README says generation is "Future" but code has production router

**Changes:**
1. [ ] Update README: Change "Future:" → "Production-ready:" for generation section
2. [ ] Add note: "Multi-provider router with circuit breaker, fallback cascades, and rate limiting implemented. Live API testing (--generation --live) not executed to avoid API costs."
3. [ ] Document current status: Mock generator + router tested, live API testing requires keys

---

## Phase 3: Test Set Construction & Rigor (1-2 weeks)

### P3.1: Expand Benchmark to 100+ Queries

**Status:** BLOCKED (needs data collection plan)

**Plan:**
1. [ ] Define query source:
   - [ ] Curate 50 additional hand-written queries (diverse domains, reasoning types)
   - [ ] Or: Extract from arXiv papers (use paper abstracts as queries, cited papers as relevant)
   - [ ] Or: Crowd-source via task (e.g., MTurk) with relevance labels

2. [ ] Define query types:
   - [ ] Lexical precision (acronyms, domain terms): 20 queries
   - [ ] Multi-hop reasoning (2-3 steps): 20 queries
   - [ ] Entity centric (who/what/where): 20 queries
   - [ ] Procedural (how-to, methods): 20 queries
   - [ ] Semantic (what, compare, contrast): 20 queries

3. [ ] Label ground truth:
   - [ ] Change from chunk ID to (document, page, answer_span) tuple
   - [ ] Allow graded relevance: exact=1.0, same-page=0.7, same-doc=0.5, irrelevant=0.0
   - [ ] Get inter-annotator agreement (Cohen's kappa ≥ 0.70)

4. [ ] Create splits:
   - [ ] Train set (70%): Use for tuning RRF k, dedup threshold, MMR λ
   - [ ] Validation set (15%): Use for early stopping, metric selection
   - [ ] Test set (15%): Use ONLY for final reported numbers, locked

---

### P3.2: Retune Hyperparameters on Train Split

**Status:** BLOCKED (needs P3.1)

**Current values** (tuned implicitly on N=22):
- DEFAULT_RRF_K=60
- DEFAULT_DEDUP_THRESHOLD=0.8
- DEFAULT_MMR_LAMBDA=0.7

**Process:**
1. [ ] Define search space:
   - RRF k ∈ {20, 40, 60, 80, 100}
   - Dedup threshold ∈ {0.6, 0.7, 0.8, 0.9}
   - MMR λ ∈ {0.5, 0.6, 0.7, 0.8, 0.9}

2. [ ] Grid search on train set only (70% of queries)
3. [ ] Evaluate on validation set (15%)
4. [ ] Report test set results (15%, never seen by tuning)

---

### P3.3: Add Statistical Significance Testing

**Status:** READY

**Changes to metrics.py:**
```python
def compute_metric_with_ci(results, metric_name, n_bootstrap=1000):
    """Compute metric + 95% bootstrap confidence interval."""
    scores = [r[metric_name] for r in results]
    ci_lower = np.percentile(scores, 2.5)
    ci_upper = np.percentile(scores, 97.5)
    mean = np.mean(scores)
    return {
        "value": mean,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }

def paired_wilcoxon_test(strategy_a_results, strategy_b_results):
    """Paired Wilcoxon signed-rank test for strategy comparison."""
    from scipy.stats import wilcoxon
    
    diff = [a['mrr'] - b['mrr'] for a, b in zip(strategy_a_results, strategy_b_results)]
    stat, pval = wilcoxon(diff)
    return {
        "statistic": stat,
        "p_value": pval,
        "significant_at_0.05": pval < 0.05,
    }
```

**Reporting:**
```
6. RRF (k=60):         MRR=0.514 [0.480, 0.548]  vs BM25: p=0.23 (not significant)
7. RRF + Dedup:        MRR=0.477 [0.443, 0.511]  vs BM25: p=0.06 (not significant)
8. RRF + Dedup + MMR:  MRR=0.486 [0.452, 0.520]  vs BM25: p=0.13 (not significant)
```

---

## Phase 4: Ground Truth & Labeling (1-2 weeks)

### P4.1: Investigate label_chunks.py Methodology

**Status:** BLOCKED (script location unknown)

**Task:** Find and document how chunk IDs were selected as ground truth

**Questions:**
1. [ ] Was lexical matching used? (Would bias toward BM25)
2. [ ] Were multiple valid chunks per query considered?
3. [ ] Who did labeling? (Single annotator? Multiple?)
4. [ ] What was the process? (Manual reading? Heuristic?)

---

### P4.2: Redesign Ground Truth to Support Graded Relevance

**Status:** READY (after P4.1)

**Current:**
```python
{
    "query": "How does the Binding Constraint Thesis affect harness comparisons?",
    "target_doc": "2605.23950v1.pdf",
    "target_chunk_idx": 1561,  # BINARY: either this chunk or not
}
```

**Proposed:**
```python
{
    "query": "How does the Binding Constraint Thesis affect harness comparisons?",
    "relevant_passages": [
        {
            "doc": "2605.23950v1.pdf",
            "page": 4,
            "span": "The Binding Constraint Thesis asserts that, in this regime, HV is often comparable to or larger than MV",
            "relevance": 1.0,  # exact answer
        },
        {
            "doc": "2605.23950v1.pdf",
            "page": 4,
            "span": "Some report results under their own harness, compounding rather than resolving attribution",
            "relevance": 0.7,  # supporting but not direct answer
        },
        {
            "doc": "2605.23950v1.pdf",
            "page": 1,
            "span": "We formalize and defend the Binding Constraint Thesis",
            "relevance": 0.5,  # same document, different section
        },
    ],
    "is_multihop": False,
}
```

**Metric updates:**
- Instead of binary is_relevant(), use span-matching with graded relevance
- NDCG@5 naturally supports graded relevance
- MRR benefits from finding exact answers (1.0) rather than supporting passages (0.5-0.7)

---

## Phase 5: Dense Model & Reranker Improvements (1-2 weeks)

### P5.1: Add Modern Dense Retrievers

**Status:** READY

**Goal:** Test whether modern embedders beat all-MiniLM

**Candidates:**
1. BGE-small-en-v1.5 (SOTA for general retrieval, 384-dim)
2. E5-small-v2 (contrastive learning, 384-dim)
3. GTE-small (multilingual, 384-dim)
4. Jina embeddings v2 (8k context, 512-dim)

**Implementation:**
1. [ ] Create `src/retrieval/retrievers/gte.py` (generic bi-encoder loader)
2. [ ] Update `RetrievalPipeline.__init__()` to support multiple dense models
3. [ ] Add strategy dispatcher entries:
   - `bge_small`: BGE-small-en-v1.5
   - `e5_small`: E5-small-v2
   - `gte_small`: GTE-small

**Testing:**
```bash
python run_eval.py  # Will now include all 3 dense models
```

**Expected:** At least one of BGE/E5/GTE should close the gap to BM25 on technical jargon.

---

### P5.2: Add Stronger Reranker

**Status:** READY

**Goal:** Test whether modern rerankers beat ms-marco-MiniLM

**Candidates:**
1. bge-reranker-v2-m3 (state-of-art, 130M params)
2. jina-reranker-v2-base (8k context, supports longer passages)
3. rankgpt-3-turbo (Claude-based, but requires API key)

**Implementation:**
1. [ ] Create `src/retrieval/retrievers/bge_reranker.py`
2. [ ] Update `RetrievalPipeline.__init__()`
3. [ ] Add strategy dispatcher:
   - `bge_reranker`: BGE-reranker-v2-m3 (50 candidate pool)
   - `bge_reranker_v2`: BGE-reranker-v2-m3 (100 candidate pool)

**Testing:**
```bash
python run_eval.py | grep "bge_reranker"
```

**Expected:** Modern reranker should outperform ms-marco on technical literature.

---

## Phase 6: Documentation & CI/CD (1 week)

### P6.1: Create Corpus Manifest

**Status:** READY

**File:** `corpus/MANIFEST.md`

```markdown
# Corpus Manifest

This corpus contains 11 arXiv papers on AI agents, retrieved under CC-BY license.

| # | arXiv ID | Title | Authors | License | Retrieved |
|---|----------|-------|---------|---------|-----------|
| 1 | 1604.08127v1 | Partially Observed Markov Decision Processes | [authors] | CC-BY-4.0 | 2026-09 |
| 2 | 1911.01547v2 | [title] | [authors] | CC-BY-4.0 | 2026-09 |
| ... | ... | ... | ... | ... | ... |

## Download Instructions

```bash
# Create scripts/download_corpus.sh
python scripts/download_corpus_from_arxiv.py --output corpus/
```

## License

All papers are distributed under CC-BY-4.0. Proper attribution is required if redistributing.
```

---

### P6.2: Add GitHub Actions CI/CD

**Status:** READY

**File:** `.github/workflows/test.yml`

```yaml
name: Tests & Validation

on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: pip install black flake8 mypy
      - run: black --check src tests
      - run: flake8 src tests
      - run: mypy src --ignore-missing-imports

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: pip install -e ".[retrieval]"
      - run: python tests/run_quick_validation.py
      - run: python -m pytest tests/ -v
```

---

## Timeline & Dependencies

```
Phase 1 (1-2 days):        Documentation fixes → P1 ✅ DONE
Phase 2 (3-5 days):        Deps & security → Requires P1 completion
Phase 3 (1-2 weeks):       Benchmark expansion → Requires P2.2 (pyproject)
Phase 4 (1-2 weeks):       Ground truth redesign → Can be parallel with P3
Phase 5 (1-2 weeks):       New models → Can be parallel after P3.1
Phase 6 (1 week):          CI/CD → Can be done anytime

Critical path: P1 → P2 → P3 → (Final metrics) = 5-8 weeks
```

---

## Success Criteria

| Milestone | Criterion | Owner |
|-----------|-----------|-------|
| Phase 1 Done | All contradictions resolved, 0 "claims unsupported by data" comments | Dev |
| Phase 2 Done | Pickle removed, pyproject.toml deployed, no security warnings | Dev |
| Phase 3 Done | 100+ query dataset with train/test split, bootstrap CIs reported | Data |
| Phase 4 Done | Graded relevance labels, inter-annotator agreement kappa ≥ 0.70 | Data |
| Phase 5 Done | BGE/E5/GTE/bge-reranker tested, winners documented | Dev |
| Phase 6 Done | CI passes on all PRs, corpus downloads from arXiv | DevOps |

---

## References

- Issue tracker: `CRITICAL_ISSUES_IDENTIFIED.md`
- Current benchmark: `EVALUATION_RESULTS_2026_09_18.md`
- Code review: `CODE_REVIEW_SUMMARY.md`

---

**Next Step:** Schedule P2.2 (pyproject.toml refactor) as blocking for P3.1 (100+ query expansion).
