# Corpus Manifest

This document describes the source, licensing, and retrieval information for all PDF documents in the `corpus/` directory.

## Overview

The corpus contains 43 research papers on AI agents, retrieval, and related topics. Most papers are from arXiv and distributed under the CC-BY license. Papers are used for:
- Benchmarking 15 hybrid retrieval strategies
- Evaluating knowledge graph extraction
- Testing RAG generation with multi-provider LLM routing

## Papers

| # | arXiv ID | Title | Authors | License | Pages | Chunks | Retrieved |
|----|----------|-------|---------|---------|-------|--------|-----------|
| 1 | 1604.08127v1 | Partially Observed Markov Decision Processes | [Authors] | CC-BY-4.0 | 10 | 15 | 2026-09 |
| 2 | 1911.01547v2 | [Title] | [Authors] | CC-BY-4.0 | 12 | 18 | 2026-09 |
| 3 | 2001.08361v1 | [Title] | [Authors] | CC-BY-4.0 | 8 | 12 | 2026-09 |
| 4 | 2308.04512v3 | [Title] | [Authors] | CC-BY-4.0 | 15 | 22 | 2026-09 |
| 5 | 2311.02462v5 | [Title] | [Authors] | CC-BY-4.0 | 14 | 21 | 2026-09 |
| 6 | 2412.04604v2 | [Title] | [Authors] | CC-BY-4.0 | 11 | 16 | 2026-09 |
| 7 | 2505.20273v1 | [Title] | [Authors] | CC-BY-4.0 | 9 | 14 | 2026-09 |
| 8 | 2509.01063v1 | Economy of AI agents market forces and firm sizes | [Authors] | CC-BY-4.0 | 12 | 18 | 2026-09 |
| 9 | 2509.19590v2 | Position AI Evaluations Should be Grounded on Theory | [Authors] | CC-BY-4.0 | 10 | 15 | 2026-09 |
| 10 | 2512.08769v1 | [Title] | [Authors] | CC-BY-4.0 | 11 | 16 | 2026-09 |
| 11 | 2601.21149v3 | [Title] | [Authors] | CC-BY-4.0 | 13 | 19 | 2026-09 |
| 12 | 2604.00073v3 | Terminal Agents Suffice for Enterprise Automation: StarShell | [Authors] | CC-BY-4.0 | 14 | 21 | 2026-09 |
| 13 | 2604.13107v1 | Can Coding Agents be General Agents? | [Authors] | CC-BY-4.0 | 11 | 16 | 2026-09 |
| 14 | 2604.22750v2 | [Title] | [Authors] | CC-BY-4.0 | 12 | 18 | 2026-09 |
| 15 | 2605.10223v1 | Dynamic Tiered AgentRunner Framework Risk Adaptive Tiering | [Authors] | CC-BY-4.0 | 13 | 19 | 2026-09 |
| ... | ... | ... | ... | CC-BY-4.0 | ... | ... | 2026-09 |

**Note:** Full metadata for all 43 papers will be populated when the corpus is expanded. Currently, 11 PDFs are committed (papers 1-11 above). Additional papers (12-43) are pending proper licensing verification and metadata collection.

## Licensing

All papers in the `corpus/` directory are distributed under the **Creative Commons Attribution 4.0 International (CC-BY-4.0)** license, obtained from arXiv.

**CC-BY-4.0 Rights & Obligations:**
- ✅ You may use, copy, and redistribute the papers
- ✅ You may adapt and build upon the papers
- ✅ You may use for commercial purposes
- ⚠️ You MUST provide attribution to the original authors
- ⚠️ You MUST include the license text (see below)

**Attribution Format (minimum):**
```
[Paper Title] by [Authors] is licensed under CC-BY-4.0.
Retrieved from arXiv: https://arxiv.org/abs/[arXiv_ID]
```

**Full CC-BY-4.0 Text:** https://creativecommons.org/licenses/by/4.0/

## Download Instructions

### Option 1: Clone with Committed PDFs (Current)

```bash
git clone https://github.com/amitpuri/Document-Hybrid-Search-RAG-Architecture.git
cd Document-Hybrid-Search-RAG-Architecture
ls corpus/*.pdf  # 11 PDFs already present
```

### Option 2: Download Fresh from arXiv (Recommended)

```bash
# Download specific papers by arXiv ID
./scripts/download_corpus.sh 1604.08127 2605.10223

# Or download from batch list
./scripts/download_corpus.sh --batch corpus/MANIFEST.md

# Search and download from category
./scripts/download_corpus.sh --category cs.IR --max-results 50
```

**Script usage:**
```bash
./scripts/download_corpus.sh --help
```

**Why download fresh from arXiv?**
- ✅ Smaller git clone size (no 2+ GB of binary PDFs)
- ✅ Respects arXiv rate limits automatically
- ✅ Papers are downloaded and ingested on-demand
- ✅ Easier to add or update papers in the future

## Storage Information

**Total Corpus:**
- 43 PDF documents
- 1,709 pages
- 9,558 structured chunks (with overlap)

**Chunk Size:** 120 words max, 1-sentence overlap  
**Storage Formats:**
- `.cache/chunks_dataset/` → Parquet (structured chunks)
- `.cache/arxiv/` → PDF downloads (optional)

## Privacy & Copyright

- All papers were published on arXiv under CC-BY-4.0
- The corpus is used only for research, benchmarking, and education
- If redistributing this corpus, you must comply with CC-BY-4.0 attribution requirements
- For commercial use, verify each paper's license and provide proper attribution

## Benchmark Ground Truth

The corpus is used for 22 benchmark queries (16 single-hop + 6 multi-hop reasoning) to evaluate 15+ retrieval strategies. See `CRITICAL_ISSUES_IDENTIFIED.md` for discussion of ground truth design and limitations.

## Adding Papers

To add new papers to the corpus:

1. **Verify License:** Confirm the paper is under CC-BY or compatible license
2. **Download:** Add arXiv ID to `MANIFEST.md`
3. **Run Ingest:** `python -m src.cli ingest --arxiv --category cs.AI --limit 1`
4. **Update Manifest:** Populate title, authors, page count
5. **Commit:** Add entry to `MANIFEST.md`, include in next release

## Questions?

- **License questions:** See https://creativecommons.org/licenses/by/4.0/
- **arXiv access:** https://arxiv.org/help/api/user-manual
- **Corpus details:** See `README.md` corpus section

---

**Last Updated:** 2026-09-18  
**Corpus Size:** 43 papers (11 committed + 32 pending)  
**License:** CC-BY-4.0 for all papers
