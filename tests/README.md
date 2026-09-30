# Tests Directory

This directory contains validation scripts, benchmark harnesses, and test utilities for the hybrid search system.

## 🧪 Test Suite Contents

```
tests/
├── TEST_HARNESS_GUIDE.md          # Comprehensive test execution guide
├── run_quick_validation.py        # Fast 4-strategy smoke test (< 30s)
├── run_comprehensive_validation.py # Full 14-strategy validation
├── run_benchmarks.py              # Latency and throughput profiling
├── run_generation_benchmarks.py   # Multi-provider LLM rate limiting tests
├── dump_full_snippets.py          # Diagnostic snippet extractor
├── test_arxiv_integration.py      # arXiv API integration tests
├── test_confidence_scoring.py     # C3 confidence scoring tests
├── test_minimal_validation.py     # Minimal validation for CI/CD
└── test_qdrant_integration.py     # Qdrant vector database tests
```

## 📋 Test Descriptions

### [TEST_HARNESS_GUIDE.md](TEST_HARNESS_GUIDE.md)
Comprehensive guide for executing all test harnesses, including:
- Quick validation workflow
- Comprehensive validation workflow
- Generation benchmarking
- Storage backend selection
- Interpreting results
- Troubleshooting common issues

### [run_quick_validation.py](run_quick_validation.py)
Fast smoke test for CI/CD and rapid iteration:
- Tests 4 core strategies (BM25, RRF, RRF+Dedup, RRF+Dedup+MMR)
- Validates generation with offline mock provider
- Runs in < 30 seconds
- Suitable for frequent development iterations

**Usage:**
```bash
python -m tests.run_quick_validation --storage memory
python -m tests.run_quick_validation --storage parquet
```

### [run_comprehensive_validation.py](run_comprehensive_validation.py)
Full validation across all 14 retrieval strategies:
- Evaluates all strategies on 14 ground-truth queries
- Produces detailed `VALIDATION_RESULTS.md` report
- Includes snippet comparisons and generation tests
- Supports live LLM API calls for generation testing

**Usage:**
```bash
python -m tests.run_comprehensive_validation
python -m tests.run_comprehensive_validation --comprehensive --live
```

### [run_benchmarks.py](run_benchmarks.py)
Performance profiling and latency benchmarks:
- Measures retrieval latency per strategy
- Throughput analysis (queries per second)
- Memory usage profiling
- Comparison of Parquet vs. InMemory storage backends

**Usage:**
```bash
python -m tests.run_benchmarks
```

### [run_generation_benchmarks.py](run_generation_benchmarks.py)
Multi-provider LLM generation testing:
- Tests rate limiting (RPM/TPM) enforcement
- Validates circuit breaker tripping and recovery
- Tests provider fallback chains
- Measures generation latency across providers

**Usage:**
```bash
python -m tests.run_generation_benchmarks
```

### [dump_full_snippets.py](dump_full_snippets.py)
Diagnostic tool for extracting retrieval snippets:
- Outputs full text of top-1 retrieved chunks per query
- Useful for debugging retrieval quality
- Supports strategy-specific analysis

**Usage:**
```bash
python -m tests.dump_full_snippets --strategy rrf_graph_dedup_mmr
```

### [test_arxiv_integration.py](test_arxiv_integration.py)
arXiv API integration tests:
- Validates rate limiting behavior
- Tests category filtering
- Verifies exponential backoff on HTTP 429
- Checks append-only chunk ID assignment

**Usage:**
```bash
python -m tests.test_arxiv_integration
```

### [test_confidence_scoring.py](test_confidence_scoring.py)
Per-Claim Calibrated Confidence (C3) tests:
- Validates HIGH/MEDIUM/LOW confidence thresholds
- Tests rank-based confidence scoring
- Verifies fusion score integration
- Checks lexical overlap calculation

**Usage:**
```bash
python -m tests.test_confidence_scoring
```

### [test_minimal_validation.py](test_minimal_validation.py)
Minimal validation for CI/CD pipelines:
- Ground-truth sanity check
- Basic retrieval functionality
- Storage backend connectivity
- Suitable for automated CI/CD execution

**Usage:**
```bash
python -m tests.test_minimal_validation --storage memory
```

### [test_qdrant_integration.py](test_qdrant_integration.py)
Qdrant vector database integration tests:
- Tests Qdrant connectivity
- Validates vector indexing
- Tests server-side payload filtering
- Verifies HNSW ANN retrieval

**Usage:**
```bash
python -m tests.test_qdrant_integration
```

## 🚀 Quick Start

### Run Fast Validation (Recommended for Development)
```bash
python -m tests.run_quick_validation --storage memory
```

### Run Full Validation (Comprehensive)
```bash
python -m tests.run_comprehensive_validation
```

### Run Performance Benchmarks
```bash
python -m tests.run_benchmarks
```

### Test Specific Components
```bash
# Test arXiv integration
python -m tests.test_arxiv_integration

# Test confidence scoring
python -m tests.test_confidence_scoring

# Test Qdrant integration
python -m tests.test_qdrant_integration
```

## 📊 Test Coverage

### Retrieval Strategies
- ✅ BM25 (sparse keyword)
- ✅ TF-IDF (sparse vector space)
- ✅ Linear fusion (α = 0.3, 0.5, 0.7)
- ✅ Reciprocal Rank Fusion (RRF, k=60)
- ✅ PPMI distributional semantics
- ✅ Cross-encoder re-ranking
- ✅ Sentence-Transformer (MiniLM)
- ✅ SPECTER2 scientific embeddings
- ✅ Adaptive hybrid weighting
- ✅ Knowledge graph retrieval
- ✅ Qdrant vector ANN
- ✅ Post-processing (deduplication, MMR)
- ✅ P0 ablation cells (graph-only, graph fusion, graph + dedup)

### Generation Providers
- ✅ Anthropic (Direct API)
- ✅ Anthropic (AWS Bedrock)
- ✅ OpenAI (Direct API)
- ✅ OpenAI (Azure OpenAI)
- ✅ Gemini (Direct API)
- ✅ Gemini (GCP Vertex AI)
- ✅ Offline mock generator

### Storage Backends
- ✅ InMemoryChunkStore
- ✅ ParquetChunkStore
- ✅ QdrantChunkStore

### Quality Metrics
- ✅ MRR (Mean Reciprocal Rank)
- ✅ Recall@1/3/5
- ✅ NDCG@5
- ✅ Entity Coverage
- ✅ Relation Coverage
- ✅ Confidence Scoring (C3)

## ⚙️ Test Configuration

### Storage Backend Selection
Tests support multiple storage backends:
```bash
--storage memory    # Fast in-memory storage
--storage parquet   # Columnar Parquet datasets (default)
--storage qdrant    # Vector database (requires local Qdrant)
```

### LLM Provider Selection
Generation tests can target specific providers:
```bash
--provider anthropic --route direct
--provider openai --route azure
--provider gemini --route vertex
--provider mock  # Offline local generator
```

### Environment Variables
Required for live LLM API tests:
```bash
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export GEMINI_API_KEY="your-key"
```

## 🔍 Interpreting Results

### Quick Validation Output
```
✓ BM25: MRR=0.573, Recall@1=0.429
✓ RRF: MRR=0.629, Recall@1=0.500
✓ RRF+Dedup: MRR=0.629, Recall@1=0.500
✓ RRF+Dedup+MMR: MRR=0.625, Recall@1=0.500
✓ Generation: Answer grounded in 3 citations
```

### Comprehensive Validation Output
Generates `VALIDATION_RESULTS.md` with:
- Strategy-by-strategy metric tables
- Query-level analysis
- Snippet comparisons
- Generation quality assessment
- Statistical significance testing (planned)

## 🐛 Troubleshooting

### Ground-Truth Validation Fails
**Error:** `AssertionError: Ground-truth index X does not contain expected doc 'Y.pdf'`

**Solution:** Corpus chunking has drifted. Re-run ingestion with `--force` flag:
```bash
python -m src.cli ingest --corpus corpus --force
```

### Qdrant Connection Fails
**Error:** `ConnectionError: Could not connect to Qdrant`

**Solution:** Start local Qdrant container:
```bash
docker compose up -d
```

### LLM API Rate Limits
**Error:** `RateLimitError: Too many requests`

**Solution:** The rate limiter is working correctly. Wait for cooldown or use offline mock:
```bash
python -m tests.run_comprehensive_validation --provider mock
```

## 📝 Adding New Tests

When adding new features:
1. Create a new test file following naming convention `test_<feature>.py`
2. Add tests to the appropriate harness (quick/comprehensive)
3. Update this README with test description
4. Update `TEST_HARNESS_GUIDE.md` if execution workflow changes

## 🔗 Related Documentation

- **Test Execution Guide**: [TEST_HARNESS_GUIDE.md](TEST_HARNESS_GUIDE.md)
- **Source Code**: See `src/` directory for implementation
- **Documentation**: See `docs/` directory for design documents
- **Agent Guidelines**: See `AGENTS.md` for operational guidelines
