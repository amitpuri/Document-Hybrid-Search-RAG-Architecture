# AI Agent Operational Guidelines (`AGENTS.md`)

This document serves as the operational manual, behavioral guidelines, and context reference for AI coding agents and automated systems working on the `Document-Hybrid-Search-RAG-Architecture` repository.

---

## 1. Repository Mission & Overview

`Document-Hybrid-Search-RAG-Architecture` is an enterprise-grade information retrieval and RAG codebase designed for scientific, technical, and academic PDF documentation.

The core challenge this repository solves:
> **Why pure vector search fails on technical literature:** Generic dense embeddings (e.g. MiniLM) diffuse technical acronyms (*"StarShell"*, *"POMDP"*, *"AgentRunner"*, *"Looped Flows"*) across generic semantic neighborhoods, achieving only 0.318 MRR vs BM25's 0.551 MRR. BM25 sparse matching provides exact lexical precision on domain jargon. This benchmark demonstrates that on technical literature with heavily domain-specific vocabulary, pure lexical search (BM25: 0.551 MRR) currently outperforms dense retrieval and most hybrid combinations. However, **cross-encoder re-ranking excels on multi-hop reasoning queries (0.408 MRR vs RRF's 0.345 MRR)**, suggesting a division of labor: use BM25/sparse for lexical precision, cross-encoders for semantic re-ranking on complex questions, and **Knowledge Graph augmented retrieval** for structural entity-relationship queries.

---

## 2. Directory Layout & Organization

```
Document-Hybrid-Search-RAG-Architecture/
├── src/                     # PRODUCTION CODE: Three-Pipeline Architecture
│   ├── common/              # Strongly-typed domain models & text utilities
│   ├── ingestion/           # Pipeline 1: PDF extraction, chunking, Knowledge Graph parsing, caching, storage
│   │   ├── extractors.py    # Multi-backend PDF extraction (pypdfium2, pdfplumber, pypdf)
│   │   ├── chunkers.py      # Sentence-aware & active section-header preserving chunking
│   │   ├── graph_extractor.py # Domain entity & typed relation extraction with IDF-weighted graph building
│   │   ├── graph_store.py   # NetworkX knowledge graph storage, 1-hop traversal & community detection
│   │   ├── cache.py         # SHA-256 parameter- & mtime-sensitive cryptographic disk cache
│   │   ├── storage.py       # Decoupled chunk storage abstractions (ParquetChunkStore / InMemoryChunkStore)
│   │   └── pipeline.py      # IngestionPipeline orchestrator
│   │
│   ├── retrieval/           # Pipeline 2: 15 hybrid search strategies (BM25, TF-IDF, PPMI, Neural, SPECTER2, Graph, RRF, MMR, Qdrant)
│   │   ├── retrievers/      # BM25, TF-IDF, PPMI, MiniLM, SPECTER2, Graph, Cross-Encoder
│   │   ├── fusion/          # Convex Linear Combination, Reciprocal Rank Fusion (RRF), Adaptive Hybrid
│   │   ├── postprocessing/  # Jaccard Result Deduplication, Maximal Marginal Relevance (MMR)
│   │   └── pipeline.py      # RetrievalPipeline orchestrator & unified single dispatch
│   │
│   ├── generation/          # Pipeline 3: Context building, RAG prompt templates, multi-provider LLM adapters
│   │   ├── context.py       # ContextBuilder with bracketed source provenance headers & graph context
│   │   ├── prompts.py       # Grounded instruction prompt templates preventing hallucinations
│   │   ├── base.py          # BaseGenerator abstract contract & GenerationResult type
│   │   ├── rate_limiter.py  # TokenBucket + ProviderRateLimiter for client-side rate limiting (RPM/TPM)
│   │   ├── providers/       # Multi-provider LLM adapters with deployment routes (Direct, Bedrock, Azure, Vertex)
│   │   ├── router.py        # LLMRouter with fallback chain and circuit breaker
│   │   ├── mock.py          # GroundedSynthesisGenerator (local offline citations synthesizer)
│   │   ├── factory.py       # Auto-detection generator factory (env-var & provider routing)
│   │   └── pipeline.py      # GenerationPipeline orchestrator
│   │
│   ├── evaluation/          # Quantitative IR benchmarking suite (MRR, Recall@K, NDCG@5, Entity/Relation coverage)
│   │   ├── metrics.py       # MRR, Recall@1/3/5, NDCG@5, entity_coverage, relation_coverage
│   │   ├── dataset.py       # 14 curated benchmark queries & ground-truth validation
│   │   └── harness.py       # Benchmark runner across all 15 strategies
│   │
│   ├── engine.py            # High-level HybridSearchEngine facade binding all 3 pipelines
│   ├── cli.py               # Unified CLI interface (search, ask, eval, ingest)
│   └── config.py            # System configuration & environment loading
│
├── tests/                   # Validation, Benchmarking & Testing Harnesses
│   ├── run_quick_validation.py         # Fast 4-strategy + generation smoke test (< 30s)
│   ├── run_comprehensive_validation.py # Full 14-strategy validation producing VALIDATION_RESULTS.md
│   ├── run_benchmarks.py               # Latency, throughput & performance profiling harness
│   ├── run_generation_benchmarks.py    # Multi-provider LLM rate limiting & circuit breaker tests
│   ├── dump_full_snippets.py           # Diagnostic top-1 retrieval snippet extractor
│   └── TEST_HARNESS_GUIDE.md           # Step-by-step test harness execution & configuration manual
│
├── updates/                 # SESSION DOCUMENTATION: Issues, roadmaps, status reports, code reviews
│   ├── README.md            # Navigation guide for all session documentation
│   ├── CRITICAL_ISSUES_IDENTIFIED.md   # 15 critical issues (Tier 1-5), exit criteria, timeline
│   ├── IMPLEMENTATION_ROADMAP.md       # 6-phase implementation plan (Phases 1-6, 5-8 weeks)
│   ├── CODE_REVIEW_SUMMARY.md          # Code quality fixes (return types, duplicate code)
│   ├── STATUS_REPORT_2026_09_18.md     # Comprehensive session status (Phases 1-2 complete)
│   ├── ROADMAP_COMPLETION_STATUS.md    # docs/roadmap.md progress (~32% complete)
│   ├── PHASE_2_COMPLETION.md           # Phase 2 details (dependencies, CI/CD, licensing)
│   ├── WORK_SUMMARY_2026_09_18.md      # Detailed session log (commits, findings, validation)
│   └── EVALUATION_RESULTS_2026_09_18.md# Benchmark results (22 queries × 18 strategies)
│
├── corpus/                  # 44 Research PDF files (1,709 pages, 9,558 structured chunks)
│   └── MANIFEST.md          # Corpus licensing (CC-BY-4.0), downloads, attribution
│
├── .cache/                  # SHA-256 parameter- & state-sensitive disk cache (.parquet, .pkl, .npy)
├── .github/workflows/       # GitHub Actions CI/CD pipelines
│   └── ci.yml               # Automated lint, test, security, docs validation
│
├── run_eval.py              # Root convenience script to run standard, comprehensive, or generation benchmarks
├── pyproject.toml           # Modern Python packaging with optional dependencies
├── .pre-commit-config.yaml  # Local code quality hooks (Black, Flake8, isort, mypy)
├── VALIDATION_RESULTS.md    # Latest benchmark & multi-strategy validation report
├── requirements.txt         # Legacy project dependencies (use pyproject.toml instead)
├── initial-approach/        # ⚠️ HISTORICAL REFERENCE - READ-ONLY. Frozen prototype code for historical comparison.
├── improved-approach/       # ⚠️ HISTORICAL REFERENCE - READ-ONLY. Intermediate prototype code for historical comparison.
└── AGENTS.md                # This operational manual for AI coding agents
```

> [!IMPORTANT]
>
> - All **new features, enhancements, and production code** must be implemented in `src/`.
> - All **validation scripts, latency benchmarks, and test harnesses** belong in `tests/`.
> - Never hardcode sensitive API keys, credentials, or personal workstation paths into code or tests.
> - **Historical reference directories (`initial-approach/` and `improved-approach/`) are READ-ONLY**. These contain frozen prototype code preserved for historical comparison and must never be modified. Do not perform or suggest any operations on these directories.
> - **Session documentation** (code reviews, implementation roadmaps, status reports) is organized in the `updates/` folder. See `updates/README.md` for navigation.

---

## 2a. Session Documentation (`updates/` folder)

All session-level documentation—code reviews, critical issues, implementation roadmaps, and status reports—is organized in the `updates/` folder for a cleaner root directory structure.

### Quick Navigation

**For understanding current state:**
- `updates/README.md` — Index and quick navigation guide
- `updates/STATUS_REPORT_2026_09_18.md` — Executive summary (Phases 1-2 complete, quality 6.3/10)
- `updates/CRITICAL_ISSUES_IDENTIFIED.md` — 15 critical issues (5 severity tiers, exit criteria)

**For planning next work:**
- `updates/IMPLEMENTATION_ROADMAP.md` — 6-phase plan (Phases 1-6, 5-8 week timeline)
- `updates/ROADMAP_COMPLETION_STATUS.md` — Roadmap progress (~32% complete)
- `updates/PHASE_2_COMPLETION.md` — Infrastructure added (dependencies, CI/CD, licensing)

**For code quality context:**
- `updates/CODE_REVIEW_SUMMARY.md` — Fixes applied this session
- `updates/EVALUATION_RESULTS_2026_09_18.md` — Benchmark validation (22 queries, 18 strategies)

---

## 2. Historical Reference Directories (READ-ONLY)

The `initial-approach/` and `improved-approach/` directories contain **frozen prototype code** that is preserved for historical comparison and development evolution tracking. These directories are **READ-ONLY** and must never be modified by agents or automated systems.

### Strict Rules for Historical Directories:

- **NO MODIFICATIONS**: Never edit, delete, or add files to these directories
- **NO OPERATIONS**: Do not perform file operations, code analysis, refactoring, or any automated processing on these directories
- **NO SUGGESTIONS**: Do not suggest changes, improvements, or operations on the historical code
- **REFERENCE ONLY**: These directories are preserved solely for understanding the evolution of the project architecture
- **COMPARISON PURPOSE**: Use these directories to compare current implementation against earlier approaches

The current production implementation is in `src/` with the three-pipeline architecture. Historical approaches were development milestones that have been superseded by the current system. Any development work should focus on the `src/` directory and related production components.

---

## 3. The Three-Pipeline Architectural Pattern

Agents modifying `src/` must preserve the strict segregation of the three pipelines:

```
Raw PDFs ──> [ Pipeline 1: Ingestion ] ──> Chunk Store & Knowledge Graph
                                                            │
Query    ──> [ Pipeline 2: Retrieval ] <────────────────────┘
                    │
            Ranked Chunks (Top-K)
                    │
Query    ──> [ Pipeline 3: Generation ] ──> Grounded Answer with Citations
```

1. **Ingestion Pipeline (`src/ingestion`)**:
   - Must remain decoupled from retrieval algorithms.
   - Preserves sentence boundaries and tracks active section headings via regex.
   - Cache keys must be generated using `compute_cache_key()` incorporating corpus directory path, chunking params, file sizes, and mtimes.
   - Storage abstractions (`BaseChunkStore`, `InMemoryChunkStore`, `ParquetChunkStore`) must be maintained so storage can be swapped for big-data platforms (Parquet/Spark/BigQuery/Vector DBs).
   - Knowledge Graph extraction (`graph_extractor.py`, `graph_store.py`) must extract domain entities/relations with corpus-IDF weighting and maintain a persistent NetworkX graph index.

2. **Retrieval Pipeline (`src/retrieval`)**:
   - All 15 strategies must dispatch through `get_strategy_rankings(query)` to maintain identical candidate pools across benchmarks.
   - **Cross-Encoder Architecture Rule**: The cross-encoder must re-rank a *wide, un-deduplicated* pool of 50 candidates (`rrf_wide[:50]`), and apply deduplication *after* scoring. Never pre-truncate candidate pools before cross-encoding.
   - **Diversity Re-Ranking Rule**: MMR must use cosine similarity against selected document vectors and balance topical relevance ($\lambda=0.7$).
   - **Knowledge Graph Retrieval Rule**: Strategy 14 (`rrf_graph_dedup_mmr`) must combine BM25, TF-IDF, and `GraphRetriever` (1-hop traversal with degree/IDF activation and Louvain community detection fallback) via calibrated RRF, followed by sliding-window Jaccard deduplication and MMR.

3. **Generation Pipeline (`src/generation`)**:
   - Must output structured `GenerationResult` objects containing exact `citations` (`DocumentChunk` instances).
   - All prompt templates must compel source citation via bracketed identifiers (`[Source N: doc | Page P | § Section]`).
   - Generators must implement `BaseGenerator`.
   - **Multi-Provider Resilience Rule**: Generators must route through `LLMRouter` with TokenBucket rate limiting (RPM/TPM tracking) and circuit breaker protection across deployment routes (Direct, AWS Bedrock, Azure OpenAI, GCP Vertex AI).
   - When external LLMs are unavailable, offline local citation synthesis must gracefully fall back to `GroundedSynthesisGenerator`.

---

## 4. Ground-Truth Sanity & Assertion Rules

> [!CAUTION]
> **Never suppress the ground-truth sanity check.**

The evaluation dataset (`src/evaluation/dataset.py`) contains 14 curated queries with exact chunk index labels (`target_chunk_idx`).

- `validate_ground_truth()` asserts that each index is within bounds and contains the target document name.
- If re-chunking parameters (`max_words`, `overlap_sentences`) are altered, chunk indices **will drift**.
- If drift occurs, agents must **fail loudly** (`AssertionError`) rather than silently producing misleading zero metrics, and re-label ground truth targets in `src/evaluation/dataset.py`.

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
   - `MetricScores` includes `entity_coverage` and `relation_coverage` alongside IR metrics (MRR, Recall@K, NDCG@5).
   - Avoid passing raw untyped tuples across pipeline boundaries.

3. **Dependencies**:
   - Sparse search: `rank-bm25` (`BM25Okapi`).
   - Dense TF-IDF: `scikit-learn` (`TfidfVectorizer(sublinear_tf=True)`).
   - Neural models: `sentence-transformers` (`all-MiniLM-L6-v2`, `cross-encoder/ms-marco-MiniLM-L-6-v2`).
   - Graph modeling: `networkx`.
   - Storage backend: `pyarrow` (Parquet datasets).
   - PDF extraction: `pypdfium2` (preferred), fallback to `pdfplumber` / `pypdf`.

---

## 5b. CI/CD Linting & Code Quality (Before Pushing)

**IMPORTANT:** All code quality checks (Black, Flake8, mypy) must pass locally BEFORE pushing to avoid CI/CD failures.

### Running Linting Locally (Before `git push`)

```bash
# Format code with Black (mutates files in-place)
black src tests

# Check code formatting without fixing
black --check src tests

# Lint with Flake8 (shows errors, does not fix)
flake8 src tests --max-line-length=100 --extend-ignore=E203

# Type checking with mypy
mypy src --ignore-missing-imports --warn-unused-ignores
```

### Common Flake8 Errors & How to Fix Locally

| Error Code | Issue | Fix |
|---|---|---|
| **F401** | Unused imports | Remove the import statement |
| **E402** | Module-level import after code | Move all `import` / `from X import` statements to top of file (after docstring, before other code) |
| **E501** | Line too long (>100 chars) | Break into multiple lines using parentheses or implicit line continuation |
| **F541** | f-string missing placeholders | Remove `f` prefix: `f"text"` → `"text"` |
| **E741** | Ambiguous variable name | Rename `l` → `left`, `O` → `obj`, etc. |

### Pre-Commit Hook (Optional Local Automation)

The repo includes `.pre-commit-config.yaml` which automatically runs linters on `git commit`:

```bash
# Install pre-commit hooks (one-time setup)
pip install pre-commit
pre-commit install

# Run all hooks on all files (test before committing)
pre-commit run --all-files
```

Once installed, hooks will run on every `git commit` and block commits that fail Black/Flake8/mypy.

### When CI/CD Linting Fails

1. **Pull latest CI failure logs** using `gh run view <RUN_ID> --log`
2. **Fix issues locally** using Black/Flake8/mypy commands above
3. **Commit fixes** with message: `"Fix linting errors: [error type + file count]"`
4. **Push and verify** the next CI/CD run passes

**Do NOT:**
- Push code that fails local linting checks (wastes CI resources and slows development)
- Try to fix linting errors **in the CI/CD pipeline itself** (defeats the purpose of pre-commit checks)
- Commit code that violates these conventions (they prevent production bugs and maintainability regressions)

---

## 6. Standard Commands Reference

### Run Quantitative Benchmark (All 14 Queries × 15 Strategies)

```bash
# Standard console evaluation
python run_eval.py
# or:
python -m src.cli eval

# Comprehensive report generation (outputs VALIDATION_RESULTS.md)
python run_eval.py --comprehensive

# Generation layer benchmarks (rate limiting, circuit breaker, fallback routing)
python run_eval.py --generation

# Fast 4-strategy test harness
python -m tests.run_quick_validation --storage memory
```

### Interactive Multi-Strategy Document Search

```bash
# Search using top-performing Knowledge Graph strategy
python -m src.cli search "POMDP belief state filtering" --strategy rrf_graph_dedup_mmr --top-k 5

# Search using classic RRF + Dedup + MMR
python -m src.cli search "POMDP belief state filtering" --strategy rrf_dedup_mmr --top-k 5
```

### Grounded RAG Question Answering

```bash
# Default grounded QA (offline mock or auto-detected API key)
python -m src.cli ask "How does the Binding Constraint Thesis affect harness comparisons?" --strategy rrf_graph_dedup_mmr --graph-mode local

# Multi-provider route selection
python -m src.cli ask "..." --provider anthropic --route direct
python -m src.cli ask "..." --provider openai --route azure
python -m src.cli ask "..." --provider gemini --route vertex
```

### Corpus Ingestion

```bash
# Ingest corpus into default Parquet partitioned store
python -m src.cli ingest --corpus corpus

# Force rebuild entire dataset and knowledge graph index
python -m src.cli ingest --corpus corpus --force
```

---

## 7. Current Benchmark Reference (Verified Ground Truth)

When evaluating or making changes, ensure metrics do not regress from these baseline numbers (22 queries: 16 single-hop + 6 multi-hop, 9,558 chunks, 44 PDFs). Current evaluation shown below uses 14 queries on 11 PDFs (2,072 chunks) — expansion to full 22-query + statistical rigor is tracked in CRITICAL_ISSUES_IDENTIFIED.md Issue 1.1. Note: Strategies 16-18 are ablation cells for isolating Graph, Dedup, and MMR contributions:

| Strategy | MRR | Recall@1 | Recall@3 | Recall@5 | NDCG@5 | Key Characteristic |
| --- | --- | --- | --- | --- | --- | --- |
| **1. Pure BM25 (Sparse)** | 0.573 | 0.429 | 0.714 | 0.786 | 0.612 | Exceptional keyword precision on domain jargon |
| 2. Pure TF-IDF (Sparse Vector Space) | 0.392 | 0.143 | 0.500 | 0.714 | 0.448 | Sparse vector space baseline with sublinear term frequencies |
| 3. Linear Hybrid (α=0.3) | 0.554 | 0.357 | 0.714 | 0.786 | 0.595 | Strong sparse bias |
| 4. Linear Hybrid (α=0.5) | 0.524 | 0.286 | 0.714 | 0.857 | 0.599 | Equal convex score combination |
| 5. Linear Hybrid (α=0.7) | 0.488 | 0.214 | 0.714 | 0.857 | 0.573 | Dense-heavy blend degraded by dense noise |
| 6. RRF (k=60) | 0.629 | 0.500 | 0.714 | 0.857 | 0.678 | Immune to score-scale distortion |
| 7. RRF + Deduplication | 0.629 | 0.500 | 0.714 | 0.857 | 0.678 | Eliminates sliding-window redundant chunks |
| **8. RRF + Dedup + MMR** | **0.625** | **0.500** | **0.786** | **0.857** | **0.683** | Top non-graph diversity via MMR (lambda=0.7) |
| 9. PPMI Semantic + BM25 RRF | 0.402 | 0.214 | 0.500 | 0.643 | 0.437 | Distributional semantics from scratch |
| 10. Cross-Encoder Re-rank | 0.483 | 0.286 | 0.571 | 0.857 | 0.567 | Re-ranks 50 un-deduplicated candidates |
| 11. Sentence-Transformer (MiniLM) | 0.292 | 0.143 | 0.357 | 0.571 | 0.339 | Pure dense bi-encoder; diffuses rare coined terms |
| 12. Adaptive Hybrid | 0.494 | 0.286 | 0.571 | 0.786 | 0.549 | Dynamic query-intent alpha weighting |
| 13. SPECTER2 (Scientific Bi-Encoder) | 0.112 | 0.000 | 0.143 | 0.286 | 0.142 | Pure dense SciBERT bi-encoder with dual asymmetric adapters ([PRX] & [QRY]) |
| **14. RRF + Graph + Dedup + MMR** | 0.565 | 0.429 | 0.714 | 0.786 | 0.621 | Fuses IDF-weighted NetworkX KG into RRF; highest relation coverage (0.643) |
| 15. Qdrant Vector (ANN) | 0.292 | 0.143 | 0.357 | 0.571 | 0.339 | Vector retrieval via Qdrant HNSW index (MiniLM dense embeddings) |
| 16. Ablation: Graph only | N/A | N/A | N/A | N/A | N/A | Isolates graph retrieval signal without BM25/TF-IDF fusion |
| 17. Ablation: RRF + Graph (no dedup/MMR) | N/A | N/A | N/A | N/A | N/A | Isolates raw graph fusion contribution |
| 18. Ablation: RRF + Graph + Dedup (no MMR) | N/A | N/A | N/A | N/A | N/A | Isolates MMR's marginal contribution on top of graph fusion |

**Footnotes:**
- **Strategy 10 (Cross-Encoder Re-rank)**: The cross-encoder (ms-marco-MiniLM-L-6-v2) is trained on MS MARCO web-passage Q&A data. On scientific/technical PDF corpora, this domain mismatch causes cross-encoder reranking (MRR=0.483) to underperform the upstream RRF retriever (MRR=0.629). This is expected behavior, not a bug. See docs/researchpaper.md Section 6.6 for detailed empirical evidence. Practitioners should use domain-adapted rerankers (e.g., scientific-papers-trained models) when deploying on specialized corpora.

- **Modern Dense Embeddings (Available, Not Yet Benchmarked)**: Two state-of-the-art dense retrievers are available in the codebase but not yet included in the 15-strategy benchmark:
  - **BGE-small-en-v1.5**: BAAI's Billion-scale General Embeddings, achieves top MTEB rankings for general retrieval. Fast (384 dims) and suitable for jargon-dense technical corpora. Implemented as `src/retrieval/retrievers/neural.py::BGERetriever`.
  - **E5-small-v2**: Contrastive learning-trained dense retriever with strong generalization across domains. Uses query/passage instruction prefixes. Implemented as `src/retrieval/retrievers/neural.py::E5Retriever`.
  - **Installation**: `pip install ".[modern-embeddings]"` (requires `sentence-transformers>=2.2.0`).
  - **Benchmarking Timeline**: Phase 3 evaluation planned. Early experiments encouraged; contributions to test these models on the 22-query benchmark are welcome.

