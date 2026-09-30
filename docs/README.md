# Documentation Directory

This directory contains research papers, design documents, and technical documentation for the hybrid search system.

## 📚 Documentation Contents

```
docs/
├── MODERN_EMBEDDINGS.md      # Overview of modern embedding techniques
├── kg-rag-design.md          # Knowledge Graph RAG design document
├── local-linting-guide.md    # Local development linting guidelines
├── parking-lot.txt           # Deferred ideas and discussion topics
├── researchpaper.md          # Full academic research paper
├── retrieval-strategies.md   # Detailed analysis of retrieval strategies
└── roadmap.md                # Project roadmap and future plans
```

## 📄 Document Descriptions

### [MODERN_EMBEDDINGS.md](MODERN_EMBEDDINGS.md)
Comprehensive overview of modern embedding techniques for information retrieval, including:
- Dense vs. sparse embeddings
- Transformer-based embeddings (BERT, MiniLM, SPECTER2)
- Cross-encoder vs. bi-encoder architectures
- Domain-adapted embeddings for scientific literature
- Embedding evaluation methodologies

### [kg-rag-design.md](kg-rag-design.md)
Design document for Knowledge Graph Augmented Retrieval (GraphRAG), covering:
- Entity extraction and relation discovery
- IDF-weighted graph construction
- 1-hop traversal and neighborhood activation
- Louvain community detection for disconnected queries
- Integration with rank fusion (RRF)
- Structural coverage metrics

### [local-linting-guide.md](local-linting-guide.md)
Local development and code quality guidelines:
- Black formatting configuration
- Flake8 linting rules
- mypy type checking setup
- Pre-commit hooks
- CI/CD integration

### [parking-lot.txt](parking-lot.txt)
Deferred ideas and discussion topics for future consideration:
- Potential features and enhancements
- Alternative approaches considered
- Research questions to explore
- Ideas requiring further investigation

### [researchpaper.md](researchpaper.md)
Full academic research paper on hybrid retrieval strategies:
- Abstract and introduction
- Methodology and experimental setup
- Quantitative benchmark results (MRR, Recall@K, NDCG@5)
- Ablation analysis of RRF → Dedup → MMR pipeline
- Knowledge graph coverage evaluation
- Query-level analysis
- Threats to validity
- References to foundational IR/NLP literature

### [retrieval-strategies.md](retrieval-strategies.md)
Detailed analysis of all 15 + 3 retrieval strategies:
- BM25 and TF-IDF sparse retrieval
- Dense neural bi-encoders (MiniLM, SPECTER2)
- Cross-encoder re-ranking
- Linear score fusion (α = 0.3, 0.5, 0.7)
- Reciprocal Rank Fusion (RRF, k=60)
- Adaptive hybrid weighting
- PPMI distributional semantics
- Knowledge graph augmented retrieval
- Qdrant vector ANN retrieval
- Post-processing (deduplication, MMR)
- P0 ablation cells (graph-only, graph fusion, graph + dedup)

### [roadmap.md](roadmap.md)
Project roadmap and implementation timeline:
- Phase 1: Core ingestion and retrieval infrastructure ✅
- Phase 2: Multi-provider LLM integration ✅
- Phase 3: Knowledge graph extraction and retrieval ✅
- Phase 4: arXiv integration and corpus growth ✅
- Phase 5: Evaluation suite expansion (planned)
- Phase 6: Production deployment and monitoring (planned)

## 🔍 Quick Navigation

**For understanding retrieval strategies:**
- Start with [retrieval-strategies.md](retrieval-strategies.md) for strategy-by-strategy analysis
- Read [researchpaper.md](researchpaper.md) for empirical validation and results

**For understanding knowledge graphs:**
- Read [kg-rag-design.md](kg-rag-design.md) for GraphRAG architecture and design decisions

**For development guidelines:**
- Follow [local-linting-guide.md](local-linting-guide.md) for code quality standards
- See [roadmap.md](roadmap.md) for project priorities and timeline

**For background on embeddings:**
- Review [MODERN_EMBEDDINGS.md](MODERN_EMBEDDINGS.md) for embedding theory and techniques

## 📊 Documentation Statistics

- **Total Documents**: 7
- **Research Papers**: 1
- **Design Documents**: 2
- **Technical Guides**: 2
- **Planning Documents**: 2

## 🔄 Documentation Updates

Documentation is updated as features are implemented and research progresses. The roadmap in [roadmap.md](roadmap.md) reflects the current implementation status and future plans.

## 📝 Contributing to Documentation

When adding new features or conducting research:
1. Update relevant design documents with architectural decisions
2. Document empirical findings in [researchpaper.md](researchpaper.md) or separate analysis documents
3. Update [roadmap.md](roadmap.md) with completed milestones
4. Add deferred ideas to [parking-lot.txt](parking-lot.txt)

## 🔗 Related Documentation

- **Source Code**: See `src/` directory for implementation
- **Test Harnesses**: See `tests/TEST_HARNESS_GUIDE.md` for validation procedures
- **Agent Guidelines**: See `AGENTS.md` for AI agent operational guidelines
- **Session Documentation**: See `updates/` folder for session-level reports and code reviews
