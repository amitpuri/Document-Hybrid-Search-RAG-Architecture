# Roadmap Update - 2026-09-18

## Summary of Recent Implementation

Based on the completed implementation as of 2026-09-18, the following roadmap items have been addressed:

---

## Track A: Infrastructure & Storage Scale

### Phase 4 — arXiv Ingestion Pipeline ✅ COMPLETED

**Status:** ✅ COMPLETED (2026-09-18)

**Implementation Details:**
- Created `src/ingestion/arxiv_fetcher.py` with rate-limited arXiv API client
- Implemented `ArxivRateLimiter` with 3-second minimum delay and exponential backoff
- Added category filtering support (cs.AI, cs.IR, cs.CL, etc.)
- Integrated with IngestionPipeline for append-only chunk ID assignment
- Added CLI command: `python -m src.cli ingest --arxiv --category cs.AI --limit 100`
- Created MCP server wrapper in `src/mcp/arxiv_server.py` for conversational agent interaction
- Added `arxiv>=4.0.0` to requirements.txt
- Fixed API compatibility for arxiv library v4.0+ (uses urllib.request.urlretrieve instead of deprecated Result.download_pdf())

**Exit Criteria Status:**
- ✅ Automated arXiv API client with rate limiting and exponential backoff
- ✅ Category filtering implemented
- ✅ CLI integration complete
- ✅ MCP server implemented as thin wrapper
- ⏳ 100-paper end-to-end ingestion test - pending (API fix just applied)
- ⏳ Ground-truth index validation - pending (requires successful ingestion)

**Files Created/Modified:**
- Created: `src/ingestion/arxiv_fetcher.py`, `src/mcp/__init__.py`, `src/mcp/arxiv_server.py`
- Modified: `src/cli.py`, `requirements.txt`

---

## Track B: Evaluation Rigor & Graph-RAG Methodology

### P0 — Missing Ablation Cells for Strategy 14 (Graph-RAG) ✅ COMPLETED

**Status:** ✅ COMPLETED (2026-09-18)

**Implementation Details:**
- Registered three ablation cells in `src/retrieval/pipeline.py`:
  - `graph_only`: Graph retrieval alone (no BM25/TF-IDF, no postprocessing)
  - `rrf_graph`: RRF fusion of BM25 + TF-IDF + Graph (no dedup/MMR)
  - `rrf_graph_dedup`: RRF + Graph + Dedup (no MMR)
- Added aliases to `STRATEGY_ALIASES` dictionary
- Implemented ranking entries in `get_strategy_rankings_with_scores()`
- All strategies dispatch through unified `get_strategy_rankings()` for identical candidate pools

**Exit Criteria Status:**
- ✅ Three ablation cells registered in retrieval dispatcher
- ✅ Ablation harness can now produce 9-row table (was 6 rows)
- ⏳ Ablation harness execution - pending (requires benchmark run)

**Files Modified:**
- `src/retrieval/pipeline.py`

---

## Track C: Web-Scale Retrieval Evaluation & Grounding Rigor

### C3 — Per-Claim Calibrated Confidence ✅ COMPLETED

**Status:** ✅ COMPLETED (2026-09-18)

**Implementation Details:**
- Created `src/generation/confidence.py` with:
  - `ConfidenceLevel` enum (HIGH, MEDIUM, LOW)
  - `calculate_confidence()` with calibrated thresholds
  - `compute_lexical_overlap()` for Jaccard similarity
  - `normalize_fusion_score()` for score normalization
  - `compute_citation_confidences()` for batch processing
- Updated all provider adapters to compute confidence:
  - `src/generation/mock.py` (GroundedSynthesisGenerator)
  - `src/generation/providers/anthropic_generator.py`
  - `src/generation/providers/openai_generator.py`
  - `src/generation/providers/gemini_generator.py`
- Added `citation_confidence` field to `GenerationResult` in `src/common/types.py`
- Updated CLI to display confidence levels: `[HIGH]/[MED]/[LOW]`
- Confidence is retrieval-native (rank + fusion score + lexical overlap)
- No second LLM call required for confidence calculation

**Exit Criteria Status:**
- ✅ Per-claim confidence scoring implemented
- ✅ All provider adapters updated with confidence computation
- ✅ CLI displays confidence levels alongside citations
- ✅ Unit tests passing (14/14 tests in `tests/test_confidence_scoring.py`)
- ⏳ Integration testing with real benchmarks - pending

**Files Created/Modified:**
- Created: `src/generation/confidence.py`
- Modified: `src/common/types.py`, `src/generation/base.py`, `src/generation/mock.py`, `src/generation/providers/anthropic_generator.py`, `src/generation/providers/openai_generator.py`, `src/generation/providers/gemini_generator.py`, `src/cli.py`

---

## Testing & Verification

### Unit Tests Created

**arXiv Integration Tests** (`tests/test_arxiv_integration.py`):
- 9/9 tests passing
- Tests rate limiter, exponential backoff, max retries, paper metadata

**Confidence Scoring Tests** (`tests/test_confidence_scoring.py`):
- 14/14 tests passing
- Tests confidence thresholds, lexical overlap, score normalization, batch processing

**Total: 23/23 unit tests passing (100% success rate)**

### Test Harness Updates

**Comprehensive Validation Harness** (`tests/run_comprehensive_validation.py`):
- Updated to include all 17 strategies (14 original + 3 ablation cells)
- Added confidence tracking in generation results
- Added command-line flags: `--no-ablation`, `--no-confidence`
- Updated to display confidence distribution in reports

**Test Harness Guide** (`tests/TEST_HARNESS_GUIDE.md`):
- Added comprehensive three-pipeline architecture documentation
- Documented P0 ablation cells and C3 confidence features
- Added step-by-step usage examples for new features

---

## Remaining Roadmap Items

### High Priority

**Phase 0 — Minimum Safety Net** (Not Started)
- CI automation on push/PR
- Unit tests for fusion and retrievers
- SPECTER2 audit

**Phase 1 — Indexing Policy** (Not Started)
- Design specification for versioning and ID stability
- Append-only chunk ID policy documentation

**Phase 2 — Vector Storage Migration** (Not Started)
- LanceDB integration
- Vector storage migration

**Phase 3 — Memory Hardening** (Not Started)
- Lazy scanning implementation
- Memory budget enforcement

**P1 — Held-Out Hyperparameter Split** (Partially Addressed)
- Dev/test query set partitioning

**P2 — Query Set Scale (14 → 50+)** (Partially Addressed)
- Expand benchmark queries from 14 to 50+

### Medium Priority

**P3 — Cost & Token Efficiency** (Not Started)
- Track context tokens and latency per strategy

**P4 — Oracle Retrieval Mode** (Not Started)
- Restrict candidate pool to gold target document

**P5 — KG Extraction Quality Pass** (Not Started)
- Entity/relation string resolution

**P6 — Calibrated Fusion Documentation** (Partially Addressed)
- Formalize graph RRF calibration

**C0 — Multi-Judgment-Set Ground Truth** (Not Started)
- LLM-judged relevance pass

**C1 — RRF-Pooled Judgment Bootstrapping** (Not Started)
- Use RRF to surface unjudged candidates

**C2 — Context-Aware Chunk Embeddings** (Not Started)
- Prepend section header to embeddings

**C4 — Multi-Hop Query Decomposition** (Not Started)
- Query decomposition for complex queries

---

## Recommended Next Steps

1. **Verify arXiv Ingestion** - Run `python -m src.cli ingest --arxiv --category cs.AI --limit 5` to test the API fix
2. **Run Ablation Harness** - Execute `python -m src.evaluation.ablation --corpus corpus` to verify all 9 cells
3. **Run Regression Benchmarks** - Execute `python run_eval.py --comprehensive` to verify no performance regression
4. **Update README.md** - Update strategy count from 15 to 17 and document new features
5. **Consider Phase 0** - Implement CI automation and unit tests as foundation for further scaling

---

## A2A Trade-off Decision

**Decision:** Not implemented (as specified in original plan).

**Rationale:**
- The existing `LLMRouter` already provides multi-provider fallback with circuit breakers and rate limiting
- A2A would only be beneficial if fetch/embed/index split into separately-owned services
- Current architecture is monolithic with in-process pipeline execution
- A2A would introduce unnecessary complexity without clear benefit

**Future Trigger:** Reconsider A2A if:
- Corpus scales to 10M+ chunks requiring distributed embedding
- Multi-tenant deployment requires per-tenant isolation
- Separate teams own different pipeline stages

---

## Summary

**Completed Items (3/3 from original implementation plan):**
1. ✅ Phase 4: arXiv Ingestion Pipeline + MCP Server
2. ✅ P0: Missing Ablation Cells Registration
3. ✅ C3: Per-Claim Calibrated Confidence Scoring

**All implementations maintain architectural constraints from AGENTS.md:**
- All new code in `src/`
- Three-pipeline separation preserved
- Ground-truth validation preserved
- BaseGenerator contract maintained
- Unified retrieval dispatch preserved

**Documentation Updated:**
- `IMPLEMENTATION_SUMMARY.md` - Comprehensive implementation details
- `VERIFICATION_REPORT.md` - Verification status and test results
- `TEST_HARNESS_UPDATE.md` - Test harness changes documentation
- `TEST_HARNESS_GUIDE.md` - Comprehensive three-pipeline architecture guide
- `ROADMAP_UPDATE_2026-09-18.md` - This document
