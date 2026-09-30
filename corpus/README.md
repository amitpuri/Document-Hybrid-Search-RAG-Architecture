# Corpus Directory

This directory contains the PDF documents that serve as the knowledge base for the hybrid search and RAG system.

## 📄 Corpus Contents

The corpus consists of **44 research PDF files** (1,709 pages, 9,558 structured chunks) covering topics in:

- Machine Learning and AI (agents, reinforcement learning, multi-agent systems)
- Information Retrieval (search algorithms, ranking, evaluation)
- Natural Language Processing (RAG, LLMs, transformers)
- Hybrid Search Systems (semantic search, knowledge graphs)

## 📂 File Organization

```
corpus/
├── MANIFEST.md          # Corpus licensing, download sources, and attribution
├── metadata.json        # Extracted metadata for all PDFs
├── 1604.08127v1.pdf     # arXiv papers (chronological)
├── 1911.01547v2.pdf
├── 2001.08361v1.pdf
├── ...
├── s41586-025-10014-0.pdf  # Nature papers
├── s41573-026-01496-2.pdf
├── ssrn-6097646.pdf     # SSRN papers
└── ...
```

## 📜 Licensing & Attribution

All documents in this corpus are licensed under **CC-BY-4.0** or similar open-access licenses. Detailed licensing information, download sources, and attribution requirements are documented in [MANIFEST.md](MANIFEST.md).

## 🔄 Corpus Updates

The corpus can be expanded through:

1. **Manual Addition**: Place new PDF files directly in this directory
2. **arXiv Integration**: Use the arXiv ingestion pipeline:
   ```bash
   python -m src.cli ingest --arxiv --category cs.AI --limit 100
   ```

## ⚠️ Important Notes

- **Chunking Regime**: The corpus is chunked with specific parameters (`max_words=200`, `overlap_sentences=1`). Changing these parameters will cause chunk indices to drift, requiring ground-truth re-labeling in `src/evaluation/dataset.py`.
- **Incremental Ingestion**: The ingestion pipeline tracks file modification times and sizes to only reprocess changed files.
- **Storage**: Chunks are stored in `.cache/chunks_dataset/` as partitioned Parquet files, not in this directory.

## 📊 Corpus Statistics

- **Total PDFs**: 44
- **Total Pages**: 1,709
- **Total Chunks**: 9,558
- **Average Chunks per PDF**: ~217
- **Knowledge Graph Entities**: 3,847
- **Knowledge Graph Relations**: 7,124

## 🔍 Corpus Management

### View Corpus Statistics
```bash
python -m src.cli ingest --corpus corpus --stats
```

### Force Re-Chunking
```bash
python -m src.cli ingest --corpus corpus --force
```

### Clear Cache and Rebuild
```bash
rm -rf .cache/chunks_dataset
python -m src.cli ingest --corpus corpus
```

## 📚 Document Sources

Documents are sourced from:

- **arXiv.org**: Preprint server for physics, mathematics, computer science, quantitative biology, quantitative finance, statistics, electrical engineering and systems science, and economics
- **Nature Portfolio**: High-impact scientific journals
- **SSRN**: Social Science Research Network

See [MANIFEST.md](MANIFEST.md) for detailed source URLs and publication metadata.
