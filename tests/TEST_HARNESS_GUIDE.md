# Test Harness Guide for Document Hybrid Search System

This guide provides step-by-step instructions for running the various test harnesses available in the Document Hybrid Search system, including detailed walkthroughs of the three-pipeline architecture.

## Overview

The test suite includes multiple harnesses for different validation purposes:

- **Quick Validation Harness** - Fast validation with key strategies
- **Comprehensive Validation Harness** - Full validation with README-formatted output (including P0 ablation cells and C3 confidence tracking)
- **Enhanced Benchmark Harness** - Performance and strategy testing
- **Generation Layer Benchmarks** - Multi-provider LLM testing
- **Unit Tests** - Component-level testing

## Recent Updates (2026-09-18)

- **P0 Ablation Cells:** Added three new retrieval strategies for factorial component isolation:
  - `graph_only` - Graph retrieval alone (no BM25/TF-IDF, no postprocessing)
  - `rrf_graph` - RRF fusion of BM25 + TF-IDF + Graph (no dedup/MMR)
  - `rrf_graph_dedup` - RRF + Graph + Dedup (no MMR)
- **C3 Confidence Scoring:** Added per-claim calibrated confidence tracking in generation results:
  - Confidence levels: HIGH, MEDIUM, LOW
  - Based on retrieval-native signals (rank, fusion score, lexical overlap)
  - No second LLM call required
- **Test Harness Updates:** Comprehensive validation harness now includes:
  - All 17 strategies (14 original + 3 ablation cells)
  - Confidence distribution tracking in generation results
  - Optional flags for backward compatibility (`--no-ablation`, `--no-confidence`)

---

## Three-Pipeline Architecture Overview

The Document Hybrid Search system follows a strict three-pipeline architecture:

```
Raw PDFs ──> [ Pipeline 1: Ingestion ] ──> Chunk Store & Knowledge Graph
                                                            │
Query    ──> [ Pipeline 2: Retrieval ] <────────────────────┘
                    │
            Ranked Chunks (Top-K)
                    │
Query    ──> [ Pipeline 3: Generation ] ──> Grounded Answer with Citations
```

### Pipeline 1: Ingestion (PDF → Chunks + Knowledge Graph)

**Purpose:** Extract text from PDFs, chunk into semantically meaningful segments, build knowledge graph, and store in a queryable format.

**Key Components:**

- `src/ingestion/extractors.py` - Multi-backend PDF extraction (pypdfium2, pdfplumber, pypdf)
- `src/ingestion/chunkers.py` - Sentence-aware chunking with section header preservation
- `src/ingestion/graph_extractor.py` - Domain entity/relation extraction with IDF-weighted graph building
- `src/ingestion/graph_store.py` - NetworkX knowledge graph storage with 1-hop traversal
- `src/ingestion/storage.py` - Decoupled chunk storage (ParquetChunkStore / InMemoryChunkStore)
- `src/ingestion/pipeline.py` - IngestionPipeline orchestrator

**Step-by-Step Ingestion Process:**

1. **PDF Discovery**

   ```bash
   # Ingestion pipeline scans corpus directory for PDF files
   python -m src.cli ingest --corpus corpus
   ```

   - Scans `corpus/` directory for `.pdf` files
   - Records file paths, sizes, and modification times
   - Computes SHA-256 cache key based on corpus state

2. **Cache Check**

   - Checks `.cache/` directory for existing cached results
   - Cache key includes: corpus path, chunking params, file sizes, mtimes
   - If cache hit and `--force` not set, skips re-processing

3. **PDF Text Extraction**

   - Tries pypdfium2 (preferred for speed and accuracy)
   - Falls back to pdfplumber if pypdfium2 fails
   - Falls back to pypdf as last resort
   - Extracts text page-by-page with metadata

4. **Sentence-Aware Chunking**

   - Splits text into sentences using NLP sentence boundaries
   - Groups sentences into chunks with configurable `max_words` (default: 200)
   - Preserves active section headers (e.g., "## 3.1 Architecture")
   - Applies sliding window with `overlap_sentences` (default: 1)
   - Tracks page numbers and section metadata per chunk

5. **Knowledge Graph Extraction**

   - Extracts domain entities (e.g., "POMDP", "StarShell", "AgentRunner")
   - Extracts typed relations (e.g., "StarShell implements AgentRunner")
   - Computes IDF weights for entities based on corpus frequency
   - Builds NetworkX graph with entity nodes and typed relation edges
   - Enables 1-hop traversal and community detection (Louvain)

6. **Storage**

   - **Parquet Storage (default):** Saves chunks to partitioned Parquet files

     - Metadata: chunk_id, document_name, page_number, section_header
     - Text: chunk_text content
     - Efficient for large corpora and big-data platforms

   - **Memory Storage:** Stores chunks in-memory for fast testing

     - Used with `--storage memory` flag
     - No persistence; lost after process exit

7. **Indexing**

   - Builds BM25 index from chunk texts
   - Builds TF-IDF index using scikit-learn TfidfVectorizer
   - Loads neural models (MiniLM, SPECTER2) for dense embeddings
   - Builds NetworkX graph index for graph retrieval
   - All indices saved to `.cache/` for reuse

**Ingestion Commands:**

```bash
# Standard ingestion (uses Parquet storage)
python -m src.cli ingest --corpus corpus

# Force rebuild (clears cache and re-processes all PDFs)
python -m src.cli ingest --corpus corpus --force

# Ingest with custom chunking parameters
python -m src.cli ingest --corpus corpus --max-words 150 --overlap-sentences 2

# Ingest with memory storage (for testing)
python -m src.cli ingest --corpus corpus --storage memory
```

**Ingestion Output:**

```
Loading corpus from: corpus
Found 11 PDF files
Extracting text from PDFs...
  Processing 2605.23950v1.pdf...
  Processing 2605.10223v1.pdf...
  ...
Chunking text...
  Total chunks: 2,072
Building knowledge graph...
  Entities: 342
  Relations: 189
Building indices...
  BM25 index: READY
  TF-IDF index: READY
  Neural models: LOADED
  Graph index: READY
Ingestion complete: 11 PDFs, 2,072 chunks
```

**Key Ingestion Invariants:**

- **Append-Only Chunk IDs:** New content must use monotonically increasing chunk IDs
- **Ground-Truth Preservation:** Existing chunk IDs must remain stable across re-ingestion
- **Cache Sensitivity:** Cache keys incorporate file sizes and mtimes to detect changes
- **Graph Consistency:** Knowledge graph must be rebuilt if any PDF changes

---

### Pipeline 2: Retrieval (Query → Ranked Chunks)

**Purpose:** Transform user queries into ranked lists of relevant document chunks using 17 hybrid retrieval strategies.

**Key Components:**
- `src/retrieval/retrievers/` - BM25, TF-IDF, PPMI, MiniLM, SPECTER2, Graph, Cross-Encoder
- `src/retrieval/fusion/` - Linear Combination, RRF, Adaptive Hybrid
- `src/retrieval/postprocessing/` - Jaccard Deduplication, MMR
- `src/retrieval/pipeline.py` - RetrievalPipeline orchestrator with unified dispatch

**Step-by-Step Retrieval Process:**

1. **Query Preprocessing**

   ```bash
   python -m src.cli search "POMDP belief state filtering" --strategy rrf_dedup_mmr --top-k 5
   ```

   - Normalizes query text (lowercase, remove special chars)
   - Tokenizes for sparse retrieval (BM25, TF-IDF)
   - Prepares for dense retrieval (neural embeddings)

2. **Strategy Dispatch**

   - All strategies dispatch through `get_strategy_rankings(query)` in pipeline.py
   - Ensures identical candidate pools across benchmarks
   - Strategy selection via `--strategy` flag or alias

3. **Sparse Retrieval (BM25 / TF-IDF)**

   - **BM25:** Uses rank-bm25 library with Okapi BM25 scoring

     - Exceptional keyword precision on domain jargon
     - Handles coined terms like "StarShell", "POMDP"

   - **TF-IDF:** Uses scikit-learn TfidfVectorizer with sublinear_tf=True

     - Dense vector space baseline
     - Term frequency-inverse document frequency weighting

4. **Dense Retrieval (Neural)**

   - **MiniLM (Sentence-Transformer):** Pre-trained all-MiniLM-L6-v2

     - Pure dense bi-encoder; diffuses rare coined terms
     - Good for semantic similarity but poor on technical jargon

   - **SPECTER2:** Scientific document embedding with dual asymmetric adapters

     - Domain-adapted for scientific literature
     - [PRX] adapter for documents, [QRY] adapter for queries

   - **Cross-Encoder:** Re-ranks 50 un-deduplicated candidates

     - Uses ms-marco-MiniLM-L-6-v2
     - Re-ranks wide pool before deduplication (per AGENTS.md rule)

5. **PPMI Semantic Retrieval**

   - Builds term co-occurrence matrix from scratch
   - Computes Positive Pointwise Mutual Information
   - Zero-dependency distributional semantics
   - Useful when neural models unavailable

6. **Graph Retrieval**

   - **GraphRetriever:** 1-hop traversal on NetworkX knowledge graph
   - **Entity Activation:** Query entities activate related nodes
   - **IDF Weighting:** Rare entities get higher activation scores
   - **Community Detection:** Louvain algorithm for related entity clusters
   - **Relation Traversal:** Follows typed relations (e.g., "implements", "extends")

7. **Rank Fusion**

   - **Linear Combination:** Convex combination of sparse + dense scores

     - Configurable α parameter (0.3, 0.5, 0.7)
     - Sensitive to score-scale distortion

   - **RRF (Reciprocal Rank Fusion):** Rank-based fusion immune to score-scale

     - Formula: `1 / (k + rank)` with k=60
     - Combines multiple ranking lists (BM25, TF-IDF, Graph)
     - Consistent performance across different score scales

   - **Adaptive Hybrid:** Dynamic α weighting based on query intent

     - Heuristic detection of keyword vs. semantic queries
     - Adjusts fusion weights in real-time

8. **Post-Processing**

   - **Jaccard Deduplication:** Removes sliding-window redundant chunks

     - Threshold: 0.7 Jaccard similarity
     - Applied after RRF fusion (per AGENTS.md rule)
     - Keeps highest-scoring chunk from duplicate groups

   - **MMR (Maximal Marginal Relevance):** Diversity re-ranking

     - λ=0.7 balance between relevance and diversity
     - Cosine similarity against selected document vectors
     - Prevents redundancy in top-K results

9. **Result Formatting**

   - Returns `SearchResult` objects with:

     - `chunk`: DocumentChunk instance
     - `score`: Fusion score
     - `rank`: Position in ranked list

   - CLI displays: `[Source N: doc.pdf | Page P | § Section]`

**Retrieval Strategies (17 Total):**

**Original 14 Strategies:**
1. `bm25` - Pure BM25 sparse keyword search
2. `tfidf` - Pure TF-IDF dense vector search
3. `linear_0.3` - Linear hybrid α=0.3 (sparse bias)
4. `linear_0.5` - Linear hybrid α=0.5 (equal blend)
5. `linear_0.7` - Linear hybrid α=0.7 (dense bias)
6. `rrf` - RRF fusion (k=60)
7. `rrf_dedup` - RRF + Deduplication
8. `rrf_dedup_mmr` - RRF + Dedup + MMR (top non-graph diversity)
9. `ppmi` - PPMI semantic + BM25 RRF
10. `cross_encoder` - Cross-Encoder re-rank (50 candidates)
11. `sentence_transformer` - Pure MiniLM dense
12. `adaptive` - Adaptive hybrid with dynamic α
13. `specter2` - SPECTER2 scientific embedding
14. `rrf_graph_dedup_mmr` - RRF + Graph + Dedup + MMR (benchmark ceiling)
15. `qdrant` - Qdrant ANN vector retrieval

**P0 Ablation Cells (3 New):**
16. `graph_only` - Graph retrieval alone (no BM25/TF-IDF, no postprocessing)
17. `rrf_graph` - RRF fusion of BM25 + TF-IDF + Graph (no dedup/MMR)
18. `rrf_graph_dedup` - RRF + Graph + Dedup (no MMR)

**Retrieval Commands:**

```bash
# Search with top-performing strategy
python -m src.cli search "POMDP belief state filtering" --strategy rrf_graph_dedup_mmr --top-k 5

# Search with classic RRF strategy
python -m src.cli search "POMDP belief state filtering" --strategy rrf_dedup_mmr --top-k 5

# Search with ablation cell (P0)
python -m src.cli search "POMDP belief state filtering" --strategy graph_only --top-k 5

# Search with specific storage backend
python -m src.cli search "POMDP belief state filtering" --strategy rrf --storage memory --top-k 5

# Search with custom top-K
python -m src.cli search "POMDP belief state filtering" --strategy rrf_dedup_mmr --top-k 10
```

**Retrieval Output:**

```
Query: POMDP belief state filtering
Strategy: rrf_dedup_mmr
Top-K: 5

#1 [Score: 0.852] 2605.23950v1.pdf | Page 12 | § 3.2 Belief State Filtering
  In partially observable Markov decision processes (POMDPs), belief state
  filtering is essential for maintaining accurate state estimates...

#2 [Score: 0.741] 2605.10223v1.pdf | Page 8 | § 2.1 Dynamic Tiering
  The AgentRunner framework implements adaptive belief state filtering...

#3 [Score: 0.698] s41586-025-10014-0.pdf | Page 15 | § 4.3 State Estimation
  Recent advances in POMDP solvers have improved belief state filtering...

...
```

**Key Retrieval Invariants:**

- **Unified Dispatch:** All strategies use `get_strategy_rankings(query)` for identical candidate pools
- **Cross-Encoder Rule:** Re-ranks 50 un-deduplicated candidates, dedup after scoring
- **MMR Rule:** λ=0.7 with cosine similarity (never modified)
- **Graph Retrieval Rule:** RRF + Graph fusion with IDF-weighted activation

---

### Pipeline 3: Generation (Query + Chunks → Grounded Answer)

**Purpose:** Generate grounded answers with source citations using multi-provider LLM adapters, including per-claim confidence scoring.

**Key Components:**

- `src/generation/context.py` - ContextBuilder with bracketed source provenance
- `src/generation/prompts.py` - Grounded instruction prompt templates
- `src/generation/base.py` - BaseGenerator abstract contract
- `src/generation/providers/` - Anthropic, OpenAI, Gemini adapters
- `src/generation/router.py` - LLMRouter with fallback chains
- `src/generation/mock.py` - GroundedSynthesisGenerator (offline)
- `src/generation/confidence.py` - Per-claim calibrated confidence (C3)

**Step-by-Step Generation Process:**

1. **Query and Retrieval**

   ```bash
   python -m src.cli ask "How does the Binding Constraint Thesis affect harness comparisons?" --strategy rrf_dedup_mmr
   ```

   - User query is received
   - Retrieval pipeline fetches top-K chunks (default: 3)
   - Chunks are ranked by selected strategy

2. **Context Building**

   - **ContextBuilder** formats retrieved chunks into structured context
   - Each chunk gets bracketed source header: `[Source N: doc.pdf | Page P | § Section]`
   - Chunks are ordered by rank
   - Optional graph context adds entity/relation information
   - **C3 Confidence:** Each chunk receives confidence level (HIGH/MEDIUM/LOW)

3. **Confidence Calculation (C3)**

   - **Retrieval-Native Signals:**

     - `rank`: Position in ranked list (lower = better)
     - `fusion_score`: RRF fusion score from retrieval
     - `lexical_overlap`: Jaccard similarity between query and chunk text

   - **Thresholds:**

     - **HIGH:** rank ≤ 2 AND fusion_score ≥ 0.7 AND lexical_overlap ≥ 0.3
     - **MEDIUM:** rank ≤ 4 OR (fusion_score ≥ 0.5 AND lexical_overlap ≥ 0.2)
     - **LOW:** Otherwise

   - **No Second LLM Call:** Confidence is computed purely from retrieval signals
   - **Result:** Confidence list parallel to citations (e.g., `[HIGH, MED, LOW]`)

4. **Prompt Construction**

   - **Prompt Template:** Grounded instruction template requiring citations
   - **Template Key Requirements:**

     - Answer the question using provided context
     - Cite sources using bracketed format: `[Source N]`
     - Do not hallucinate information not in context
     - Indicate uncertainty if context is insufficient

   - **Context Injection:** Formatted chunks with source headers inserted
   - **Confidence Injection:** Confidence levels displayed with chunks

5. **Provider Selection**

   - **Auto-Detection:** `factory.py` checks for API keys in order:

     1. Anthropic if `ANTHROPIC_API_KEY` set
     2. OpenAI if `OPENAI_API_KEY` set
     3. Gemini if `GEMINI_API_KEY` set
     4. GroundedSynthesisGenerator (mock) if no keys

   - **Manual Selection:** User can specify `--provider` flag
   - **Router:** `LLMRouter` handles fallback chains and circuit breakers

6. **Rate Limiting**

   - **TokenBucket:** Client-side rate limiting per provider
   - **Metrics Tracked:**

     - RPM (Requests Per Minute)
     - TPM (Tokens Per Minute)

   - **Circuit Breaker:** Automatically disables provider after consecutive failures
   - **Backoff:** Exponential backoff on rate limit errors

7. **Deployment Routes**

   - **Direct:** Direct API call to provider endpoint
   - **Bedrock:** AWS Bedrock integration (Anthropic, OpenAI)
   - **Azure:** Azure OpenAI integration
   - **Vertex:** GCP Vertex AI integration (Gemini)
   - **Route Selection:** Via `--route` flag

8. **LLM Generation**

   - **Prompt + Context → LLM:**

     - Anthropic: Claude Sonnet 5 (or configured model)
     - OpenAI: GPT-5.5 (or configured model)
     - Gemini: Gemini 3.8 Flash (or configured model)

   - **Response Processing:**

     - Extract answer text
     - Parse bracketed citations: `[Source N]`
     - Validate citations against provided context
     - Map citations back to original chunks

9. **Result Formatting**

   - **GenerationResult:** Structured output with:

     - `query`: Original user query
     - `answer`: Generated answer text
     - `citations`: List of DocumentChunk instances
     - `citation_confidence`: List of confidence levels (HIGH/MEDIUM/LOW)
     - `strategy_used`: Retrieval strategy name

   - **CLI Display:**

     - Answer text with bracketed citations
     - Confidence indicators: `[HIGH]/[MED]/[LOW]` per citation
     - Source metadata: document, page, section

**Generation Providers:**

**1. Anthropic Generator**

- Model: Claude Sonnet 5 (configurable)
- Deployment Routes: Direct, Bedrock
- Strengths: Strong reasoning, good with technical content
- Fallback: OpenAI → Gemini → Mock

**2. OpenAI Generator**

- Model: GPT-5.5 (configurable)
- Deployment Routes: Direct, Azure
- Strengths: Fast, cost-effective, good for general queries
- Fallback: Anthropic → Gemini → Mock

**3. Gemini Generator**

- Model: Gemini 3.8 Flash (configurable)
- Deployment Routes: Direct, Vertex
- Strengths: Fast inference, good for large contexts
- Fallback: Anthropic → OpenAI → Mock

**4. GroundedSynthesisGenerator (Mock)**

- Model: Offline local citation synthesizer
- Deployment: In-process, no API calls
- Strengths: Zero cost, deterministic, always available
- Behavior: Synthesizes answer by extracting relevant sentences from chunks

**Generation Commands:**

```bash
# Default generation (auto-detects provider)
python -m src.cli ask "How does the Binding Constraint Thesis affect harness comparisons?" --strategy rrf_dedup_mmr

# Specific provider with route
python -m src.cli ask "How does the Binding Constraint Thesis affect harness comparisons?" --provider anthropic --route direct --strategy rrf_dedup_mmr

# Specific model
python -m src.cli ask "How does the Binding Constraint Thesis affect harness comparisons?" --provider openai --model gpt-5.5 --strategy rrf_dedup_mmr

# Mock generation (offline)
python -m src.cli ask "How does the Binding Constraint Thesis affect harness comparisons?" --provider mock --strategy rrf_dedup_mmr

# With graph context
python -m src.cli ask "How does the Binding Constraint Thesis affect harness comparisons?" --strategy rrf_graph_dedup_mmr --graph-mode local

# Custom top-K
python -m src.cli ask "How does the Binding Constraint Thesis affect harness comparisons?" --strategy rrf_dedup_mmr --top-k 5
```

**Generation Output:**

```
Query: How does the Binding Constraint Thesis affect harness comparisons?
Strategy: rrf_dedup_mmr
Provider: anthropic (claude-sonnet-5)

================================================================================
The Binding Constraint Thesis significantly impacts harness comparisons across
different AI models by establishing that model performance is fundamentally
limited by the harness's ability to constrain the search space [Source 1: 2605.23950v1.pdf | Page 7 | § 2.3 Binding Constraints]. Specifically, when a harness
provides tighter constraints on the action space, even weaker models can achieve
comparable performance to stronger models operating under looser constraints
[Source 2: 2605.10223v1.pdf | Page 12 | § 3.1 Comparative Analysis].

This thesis implies that direct model-to-model comparisons are only meaningful
when the harness constraints are held constant across tests [Source 3: 2605.23950v1.pdf | Page 9 | § 2.5 Experimental Design].
================================================================================

[Source 1: 2605.23950v1.pdf | Page 7 | § 2.3 Binding Constraints] [HIGH]
[Source 2: 2605.10223v1.pdf | Page 12 | § 3.1 Comparative Analysis] [MED]
[Source 3: 2605.23950v1.pdf | Page 9 | § 2.5 Experimental Design] [HIGH]

Citations: 3 chunks
Confidence: 2 HIGH, 1 MEDIUM, 0 LOW
```

**Key Generation Invariants:**

- **BaseGenerator Contract:** All providers must implement `generate()` method
- **Citation Requirement:** All prompts must compel source citation via brackets
- **Confidence Consistency:** All providers compute confidence identically via shared `calculate_confidence()`
- **No Second LLM Call:** Confidence is retrieval-native, not LLM-generated
- **Graceful Fallback:** Falls back to mock if all providers fail or unavailable

---

## Quick Validation Harness

**File:** `tests/run_quick_validation.py`

**Purpose:** Fast validation using key retrieval strategies and generation providers. Generates README-formatted output for Side-by-Side Retrieval Comparison and LLM Adapter Results.

### Usage

```bash
# Quick validation with parquet storage (default)
python -m tests.run_quick_validation

# Quick validation with memory storage (faster)
python -m tests.run_quick_validation --storage memory

# Quick validation with custom output file
python -m tests.run_quick_validation --output MY_VALIDATION.md

# Quick validation with live LLM API calls
python -m tests.run_quick_validation --live
```

### What It Tests

**Ground-Truth Sanity Assertion:**

- Automatically invokes `validate_ground_truth()` on startup against the ingested corpus chunks to guard against chunking index drift

**Retrieval Strategies (4 key strategies):**

- `bm25` - Pure BM25 sparse keyword search
- `rrf` - Reciprocal Rank Fusion
- `rrf_dedup_mmr` - RRF + Dedup + MMR
- `rrf_graph_dedup_mmr` - RRF + Graph + Dedup + MMR

**Generation Providers:**
- Mock generation (always runs)
- Anthropic (if `--live` flag and API key available)
- OpenAI (if `--live` flag and API key available)
- Gemini (if `--live` flag and API key available)

**Queries:**
- Query (a): "How does the Binding Constraint Thesis affect harness comparisons?"
- Query (b): "Dynamic Tiered AgentRunner Framework Risk Adaptive Tiering"

### Expected Output

Generates a markdown file with:

- Side-by-Side Retrieval Comparison tables for both queries
- LLM Adapter Results section with provider outputs
- Validation summary with success/failure counts

---

## Comprehensive Validation Harness

**File:** `tests/run_comprehensive_validation.py`

**Purpose:** Full validation using all 17 retrieval strategies (14 original + 3 P0 ablation cells) and complete evaluation metrics with C3 confidence tracking. Generates all three README-formatted sections.

### Comprehensive Validation Usage

```bash
# Comprehensive validation with parquet storage (default)
python -m tests.run_comprehensive_validation

# Comprehensive validation with memory storage
python -m tests.run_comprehensive_validation --storage memory

# Comprehensive validation with custom output file
python -m tests.run_comprehensive_validation --output FULL_VALIDATION.md

# Comprehensive validation with live LLM API calls
python -m tests.run_comprehensive_validation --live

# Comprehensive validation without ablation cells (faster)
python -m tests.run_comprehensive_validation --no-ablation

# Comprehensive validation without confidence tracking (legacy)
python -m tests.run_comprehensive_validation --no-confidence

# Comprehensive validation with query subset
python -m tests.run_comprehensive_validation --query-preset core
python -m tests.run_comprehensive_validation --query-preset new
python -m tests.run_comprehensive_validation --query-preset all
python -m tests.run_comprehensive_validation --query-labels a,b,c,e
```

### What It Tests

**Ground-Truth Sanity Assertion:**

- Automatically invokes `validate_ground_truth()` on startup against the ingested corpus chunks to guard against chunking index drift

**All 17 Retrieval Strategies (14 Original + 3 P0 Ablation Cells):**

**Original 14 Strategies:**

- `bm25`, `tfidf`, `linear_0.3`, `linear_0.5`, `linear_0.7`
- `rrf`, `rrf_dedup`, `rrf_dedup_mmr`
- `ppmi`, `cross_encoder`, `sentence_transformer`
- `adaptive`, `specter2`, `rrf_graph_dedup_mmr`, `qdrant`

**P0 Ablation Cells (New):**

- `graph_only` - Graph retrieval alone (no BM25/TF-IDF, no postprocessing)
- `rrf_graph` - RRF fusion of BM25 + TF-IDF + Graph (no dedup/MMR)
- `rrf_graph_dedup` - RRF + Graph + Dedup (no MMR)

**Evaluation Metrics:**

- MRR (Mean Reciprocal Rank)
- Recall@1, Recall@3, Recall@5
- NDCG@5 (Normalized Discounted Cumulative Gain)
- Entity Coverage
- Relation Coverage

**Generation Providers:**

- Same as Quick Validation harness
- **C3 Confidence Tracking:** Captures HIGH/MEDIUM/LOW confidence levels per citation

### Expected Output

Generates a markdown file with:

- Empirical Benchmark Results table (all 17 strategies with metrics, including ablation cells)
- Side-by-Side Retrieval Comparison tables (all 17 strategies per query)
- LLM Adapter Results section with confidence distribution
- Validation summary

**New Output Sections:**

**Empirical Benchmark Results Table (18 rows):**
```
| # | Strategy Name | MRR | Recall@1 | Recall@3 | Recall@5 | NDCG@5 | Key Characteristic |
|---|---|---|---|---|---|---|---|
| 1 | 1. Pure BM25 (Sparse) | 0.573 | 0.429 | 0.714 | 0.786 | 0.612 | Exceptional keyword precision on domain jargon |
...
| 16 | Ablation: Graph only | 0.xxx | 0.xxx | 0.xxx | 0.xxx | 0.xxx | Ablation: Graph retrieval alone (no BM25/TF-IDF, no postprocessing) |
| 17 | Ablation: RRF + Graph (no dedup/MMR) | 0.xxx | 0.xxx | 0.xxx | 0.xxx | 0.xxx | Ablation: RRF fusion of BM25 + TF-IDF + Graph (no dedup/MMR) |
| 18 | Ablation: RRF + Graph + Dedup (no MMR) | 0.xxx | 0.xxx | 0.xxx | 0.xxx | 0.xxx | Ablation: RRF + Graph + Dedup (no MMR) |
```

**LLM Adapter Results Section (with confidence):**
```
### Anthropic — `claude-sonnet-5`

> The Binding Constraint Thesis significantly impacts harness comparisons...

*Citations: 3 chunks*
*Confidence: 2 HIGH, 1 MEDIUM, 0 LOW*
```

### New Command-Line Flags

**--no-ablation**
- Excludes P0 ablation cells from validation
- Reduces strategy count from 17 to 14
- Faster execution (no graph-only indexing required)
- Useful for quick regression testing

**--no-confidence**
- Disables C3 confidence tracking in generation results
- Does not extract or report confidence levels
- Useful for backward compatibility with older reports

**--query-preset**
- `core` (default): Queries a, b, c, d (4 queries)
- `new`: Queries c-l (10 queries, expanded literature)
- `all`: Queries a-l (12 queries, full suite)

**--query-labels**
- Comma-separated query labels to evaluate
- Example: `--query-labels a,b,c,e`
- Useful for testing specific queries

---

## Enhanced Benchmark Harness

**File:** `tests/run_benchmarks.py`

**Purpose:** Flexible benchmark harness for testing retrieval strategies and generation pipelines with performance metrics and error reporting.

### Enhanced Benchmark Usage

```bash
# Run search benchmarks (default mode)
python tests/run_benchmarks.py

# Run all benchmark types
python tests/run_benchmarks.py --mode all

# Run generation benchmarks with live API calls
python tests/run_benchmarks.py --mode generation --live

# Run full evaluation benchmark
python tests/run_benchmarks.py --mode evaluation

# Test specific strategies
python tests/run_benchmarks.py --strategies bm25 rrf rrf_dedup_mmr

# Test with specific storage backend
python tests/run_benchmarks.py --storage memory

# Export results to JSON
python tests/run_benchmarks.py --output benchmark_results.json

# Quiet mode (less verbose output)
python tests/run_benchmarks.py --quiet
```

### Benchmark Modes

- **search** - Tests retrieval strategies (default)
- **generation** - Tests generation providers
- **evaluation** - Runs full evaluation benchmark
- **all** - Runs all benchmark types

### Expected Output

Console output with:

- Performance metrics (duration, success rates)
- Error reporting for failed tests
- Results summary by strategy
- Optional JSON export for further analysis

---

## Generation Layer Benchmarks

**File:** `tests/run_generation_benchmarks.py`

**Purpose:** Specialized benchmark for testing the multi-provider LLM generation system with rate limiting and circuit breaking.

### Generation Benchmark Usage

```bash
# Run generation benchmarks (mock only)
python tests/run_generation_benchmarks

# Run with live API calls
python tests/run_generation_benchmarks --live
```

### What It Tests

- Rate limiting mechanics (TokenBucket, ProviderRateLimiter)
- Provider configuration loading
- Circuit breaker behavior
- Fallback chain routing
- Retry and backoff logic
- Live provider API calls (if `--live` flag)

### Expected Output

Console output with:

- Rate limiting benchmark results
- Provider configuration details
- Circuit breaker state transitions
- Retry logic demonstration
- Live provider test results (if enabled)

---

## Unit Tests

**Files:**

- `tests/test_parquet_storage.py` - Parquet storage tests
- `tests/test_confidence_scoring.py` - C3 confidence scoring tests (NEW)
- `tests/test_arxiv_integration.py` - arXiv ingestion integration tests (NEW)

### Unit Test Usage

```bash
# Run all unit tests (requires pytest)
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_parquet_storage.py

# Run with verbose output
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run confidence scoring tests (standalone, no pytest required)
python tests/test_confidence_scoring.py

# Run arXiv integration tests (standalone, no pytest required)
python tests/test_arxiv_integration.py
```

### What It Tests

**Parquet Storage Tests:**
- Metadata preservation
- Projection pushdown
- Batch operations
- Atomic write safety

**Confidence Scoring Tests (NEW - C3):**
- `ConfidenceLevel` enum values (HIGH, MEDIUM, LOW)
- `calculate_confidence()` threshold logic
- `compute_lexical_overlap()` Jaccard similarity
- `normalize_fusion_score()` score normalization
- `compute_citation_confidences()` batch processing
- `CitationWithConfidence` dataclass
- Edge cases: empty queries, empty chunks, perfect overlap

**arXiv Integration Tests (NEW - Phase 4):**
- `ArxivRateLimiter` initialization and configuration
- `ArxivRateLimiter.wait()` delay enforcement
- `ArxivRateLimiter.record_success()` failure reset
- `ArxivRateLimiter.record_failure()` failure increment
- `ArxivRateLimiter` exponential backoff behavior
- `ArxivRateLimiter` max retries exceeded handling
- `ArxivRateLimiter` backoff capping at max
- `ArxivRateLimiter.reset()` state clearing
- `ArxivPaper` dataclass to_dict() conversion

### Unit Test Results

**Confidence Scoring Tests:**
```
[PASS] test_enum_values
[PASS] test_high_confidence_all_conditions_met
[PASS] test_not_high_rank_too_low
[PASS] test_medium_confidence_rank_condition
[PASS] test_low_confidence_default
[PASS] test_perfect_overlap
[PASS] test_no_overlap
[PASS] test_empty_query
[PASS] test_normalization_typical_score
[PASS] test_normalization_above_max
[PASS] test_empty_chunks
[PASS] test_single_chunk_high_confidence
[PASS] test_confidence_parallel_to_chunks
[PASS] test_creation

14 tests passed, 0 tests failed
```

**arXiv Integration Tests:**
```
[PASS] test_initialization
[PASS] test_wait_on_first_call
[PASS] test_record_success_resets_failures
[PASS] test_record_failure_increments_counter
[PASS] test_max_retries_exceeded
[PASS] test_exponential_backoff
[PASS] test_backoff_capped_at_max
[PASS] test_reset
[PASS] test_to_dict

9 tests passed, 0 tests failed
```

**Total: 23/23 unit tests passing (100% success rate)**

---

## Main Evaluation Script

**File:** `run_eval.py`

**Purpose:** Main entry point for running evaluation benchmarks with comprehensive validation support, including P0 ablation cells and C3 confidence tracking.

### Main Evaluation Script Usage

```bash
# Standard retrieval evaluation (14 queries × 17 strategies with ablation cells)
python run_eval.py

# Run with specific storage backend
python run_eval.py --storage memory

# Run generation layer benchmarks
python run_eval.py --generation

# Run generation with live API calls
python run_eval.py --generation --live

# Run comprehensive validation with README-formatted output
python run_eval.py --comprehensive

# Run comprehensive validation with live LLM testing
python run_eval.py --comprehensive --live

# Run comprehensive validation without ablation cells (faster)
python run_eval.py --comprehensive --no-ablation

# Run comprehensive validation without confidence tracking
python run_eval.py --comprehensive --no-confidence
```

### When to Use Each Mode

- **Default (no flags)** - Run standard 14-query × 17-strategy evaluation (including P0 ablation cells)
- **--generation** - Test generation layer only with C3 confidence tracking
- **--comprehensive** - Generate README-formatted validation report with all features
- **--live** - Enable live API calls for generation testing
- **--no-ablation** - Exclude P0 ablation cells for faster execution
- **--no-confidence** - Disable C3 confidence tracking for legacy compatibility

---

## CLI Integration

The test harnesses integrate with the main CLI for interactive testing:

```bash
# Interactive multi-strategy document search
python -m src.cli search "POMDP belief state filtering" --strategy rrf_dedup_mmr --top-k 5

# Grounded RAG question answering
python -m src.cli ask "How does the Binding Constraint Thesis affect harness comparisons?" --strategy rrf_dedup_mmr

# With specific provider
python -m src.cli ask "..." --provider anthropic --route direct

# Corpus ingestion
python -m src.cli ingest --corpus corpus
```

---

## API Key Configuration

For live LLM testing, configure API keys:

```bash
# Set environment variables
export ANTHROPIC_API_KEY="your_anthropic_key"
export OPENAI_API_KEY="your_openai_key"
export GEMINI_API_KEY="your_gemini_key"

# Or use .env file in project root
echo "ANTHROPIC_API_KEY=your_key" > .env
echo "OPENAI_API_KEY=your_key" >> .env
echo "GEMINI_API_KEY=your_key" >> .env
```

---

## Common Workflows

### 1. Quick Validation (Recommended for Regular Testing)

```bash
# Fast validation with key strategies
python -m tests.run_quick_validation --storage memory
```

### 2. Full Validation Before Release

```bash
# Comprehensive validation with all 17 strategies (including ablation cells)
python -m tests.run_comprehensive_validation --storage parquet

# Faster validation without ablation cells
python -m tests.run_comprehensive_validation --storage parquet --no-ablation
```

### 3. Live LLM Testing with Confidence Tracking

```bash
# Test with live API keys and confidence tracking
python -m tests.run_quick_validation --live

# Comprehensive validation with live LLM and confidence
python -m tests.run_comprehensive_validation --live
```

### 4. Ablation Cell Testing (P0)

```bash
# Test graph-only retrieval
python -m src.cli search "POMDP belief state filtering" --strategy graph_only --top-k 5

# Test RRF + Graph fusion
python -m src.cli search "POMDP belief state filtering" --strategy rrf_graph --top-k 5

# Test RRF + Graph + Dedup
python -m src.cli search "POMDP belief state filtering" --strategy rrf_graph_dedup --top-k 5
```

### 5. Confidence Scoring Testing (C3)

```bash
# Test confidence display in generation
python -m src.cli ask "How does the Binding Constraint Thesis affect harness comparisons?" --provider mock --strategy rrf_dedup_mmr

# Test confidence with live provider
python -m src.cli ask "How does the Binding Constraint Thesis affect harness comparisons?" --provider anthropic --strategy rrf_dedup_mmr

# Run unit tests for confidence scoring
python tests/test_confidence_scoring.py
```

### 6. arXiv Ingestion Testing (Phase 4)

```bash
# Test arXiv rate limiter and fetcher
python tests/test_arxiv_integration.py

# Ingest small batch of arXiv papers (requires arxiv package)
pip install arxiv
python -m src.cli ingest --arxiv --category cs.AI --limit 5

# Ingest with custom query
python -m src.cli ingest --arxiv --arxiv-query "reinforcement learning" --limit 10
```

### 7. Performance Benchmarking

```bash
# Benchmark specific strategies
python tests/run_benchmarks.py --strategies bm25 rrf --output benchmark_results.json

# Benchmark ablation cells
python tests/run_benchmarks.py --strategies graph_only rrf_graph rrf_graph_dedup --output ablation_benchmarks.json
```

### 8. Unit Testing

```bash
# Run all unit tests (requires pytest)
python -m pytest tests/ -v

# Run confidence scoring tests (standalone)
python tests/test_confidence_scoring.py

# Run arXiv integration tests (standalone)
python tests/test_arxiv_integration.py
```

---

## Output Files

The validation harnesses generate markdown files in the project root:

- `VALIDATION_RESULTS.md` - Default output from validation harnesses
- Custom filenames can be specified with `--output` flag

These files contain:
- Empirical Benchmark Results tables
- Side-by-Side Retrieval Comparison tables  
- LLM Adapter Results sections
- Validation summaries

---

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Ensure you're in the project root
cd /path/to/document-hybrid-search
pip install -r requirements.txt
```

**Cache Issues:**
```bash
# Clear cache and rebuild
python -m src.cli ingest --force --corpus corpus
```

**API Key Issues:**
```bash
# Verify environment variables
echo $ANTHROPIC_API_KEY
echo $OPENAI_API_KEY
echo $GEMINI_API_KEY
```

**Storage Backend Issues:**
```bash
# Try memory storage if Parquet fails
python -m tests.run_quick_validation --storage memory
```

---

## Best Practices

1. **Use Quick Validation** for regular development testing
2. **Use Comprehensive Validation** before releases or major changes
3. **Use Memory Storage** for faster test execution during development
4. **Use Parquet Storage** for production-like validation
5. **Enable Live Mode** sparingly to avoid API quota consumption
6. **Review Generated Reports** to compare against README reference data

---

## Summary

The test harness system provides:

- **Quick Validation** - Fast validation with key strategies
- **Comprehensive Validation** - Full validation with README formatting (17 strategies including P0 ablation cells)
- **Enhanced Benchmarking** - Performance and strategy testing
- **Generation Benchmarks** - Multi-provider LLM testing with C3 confidence tracking
- **Unit Tests** - Component-level testing (23 tests including confidence and arXiv integration)

**Recent Enhancements (2026-09-18):**
- **P0 Ablation Cells:** Added graph_only, rrf_graph, rrf_graph_dedup strategies for factorial component isolation
- **C3 Confidence Scoring:** Added per-claim calibrated confidence tracking (HIGH/MEDIUM/LOW) in generation results
- **arXiv Integration:** Added Phase 4 arXiv ingestion pipeline with rate limiting and MCP server
- **Test Harness Updates:** Comprehensive validation now includes 17 strategies and confidence distribution tracking

All harnesses use real corpus data from the README and generate results in the same format as the README reference sections for easy comparison.

---

## Additional Documentation

- **TEST_HARNESS_UPDATE.md** - Detailed changelog for test harness updates (P0 and C3 integration)
- **IMPLEMENTATION_SUMMARY.md** - Comprehensive implementation details for roadmap items
- **VERIFICATION_REPORT.md** - Verification status and test results for implemented features
- **AGENTS.md** - Operational guidelines and architectural constraints for AI agents
- **docs/roadmap.md** - Maintainer's phased plan for system evolution