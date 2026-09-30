# Scripts Directory

This directory contains utility scripts for corpus management, PDF metadata extraction, and diagnostic tools.

## 🛠️ Script Contents

```
scripts/
├── diagnose_cross_encoder.py          # Cross-encoder diagnostic tool
├── download_corpus.sh                 # Corpus download automation
├── extract_pdf_metadata.py            # PDF metadata extraction
└── regenerate_corpus_from_metadata.py # Corpus regeneration from metadata
```

## 📋 Script Descriptions

### [diagnose_cross_encoder.py](diagnose_cross_encoder.py)
Diagnostic tool for cross-encoder re-ranking analysis:
- Inspects cross-encoder scoring behavior
- Validates wide-pool-then-dedup architecture
- Identifies potential domain mismatch issues
- Outputs detailed scoring diagnostics

**Usage:**
```bash
python scripts/diagnose_cross_encoder.py --query "your query" --strategy cross_encoder
```

**Purpose:**
The cross-encoder (`ms-marco-MiniLM-L-6-v2`) can underperform on domain-specific scientific literature due to training on general web text. This script helps diagnose:
- Whether scoring inversions are occurring
- If candidate pool truncation is affecting results
- Whether domain mismatch is causing precision degradation

### [download_corpus.sh](download_corpus.sh)
Automated corpus download script:
- Downloads PDFs from arXiv, Nature, and SSRN
- Uses metadata.json for source URLs
- Validates file integrity
- Organizes files in corpus/ directory

**Usage:**
```bash
bash scripts/download_corpus.sh
```

**Requirements:**
- `wget` or `curl` for downloading
- Internet connectivity
- Write permissions to corpus/ directory

**Purpose:**
Automates the initial corpus setup by downloading all PDFs referenced in `corpus/metadata.json`. This ensures reproducible corpus setup across environments.

### [extract_pdf_metadata.py](extract_pdf_metadata.py)
PDF metadata extraction utility:
- Extracts title, authors, abstract, publication date
- Identifies document type (arXiv, journal, preprint)
- Generates structured metadata JSON
- Supports batch processing

**Usage:**
```bash
python scripts/extract_pdf_metadata.py --input corpus/ --output corpus/metadata.json
python scripts/extract_pdf_metadata.py --input corpus/2605.23950v1.pdf --single
```

**Output Format:**
```json
{
  "filename": "2605.23950v1.pdf",
  "title": "Paper Title",
  "authors": ["Author 1", "Author 2"],
  "abstract": "Paper abstract...",
  "publication_date": "2024-01-15",
  "source": "arXiv",
  "arxiv_id": "2605.23950v1",
  "categories": ["cs.AI", "cs.LG"]
}
```

**Purpose:**
Creates structured metadata for corpus documents, enabling:
- Automated corpus management
- Searchable document catalog
- Citation and attribution tracking
- Integration with ingestion pipeline

### [regenerate_corpus_from_metadata.py](regenerate_corpus_from_metadata.py)
Corpus regeneration utility:
- Rebuilds corpus from metadata.json
- Validates file presence and integrity
- Reports missing or corrupted files
- Supports selective regeneration

**Usage:**
```bash
python scripts/regenerate_corpus_from_metadata.py --metadata corpus/metadata.json
python scripts/regenerate_corpus_from_metadata.py --metadata corpus/metadata.json --check-only
```

**Purpose:**
Useful for:
- Verifying corpus integrity after manual changes
- Re-downloading missing files
- Synchronizing corpus across environments
- Auditing corpus composition

## 🚀 Common Workflows

### Initial Corpus Setup
```bash
# 1. Download corpus from sources
bash scripts/download_corpus.sh

# 2. Extract metadata
python scripts/extract_pdf_metadata.py --input corpus/ --output corpus/metadata.json

# 3. Verify corpus integrity
python scripts/regenerate_corpus_from_metadata.py --metadata corpus/metadata.json --check-only

# 4. Ingest corpus into chunk store
python -m src.cli ingest --corpus corpus
```

### Diagnose Retrieval Issues
```bash
# 1. Run cross-encoder diagnostics
python scripts/diagnose_cross_encoder.py --query "POMDP belief state" --strategy cross_encoder

# 2. Review diagnostic output
# 3. Adjust retrieval strategy or parameters based on findings
```

### Corpus Maintenance
```bash
# 1. Check corpus integrity
python scripts/regenerate_corpus_from_metadata.py --metadata corpus/metadata.json --check-only

# 2. Update metadata for new PDFs
python scripts/extract_pdf_metadata.py --input corpus/new_papers/ --append corpus/metadata.json

# 3. Re-ingest with new documents
python -m src.cli ingest --corpus corpus
```

## 🔍 Script Dependencies

### diagnose_cross_encoder.py
- `src.retrieval` (RetrievalPipeline)
- `src.ingestion` (IngestionPipeline)
- `sentence-transformers` (cross-encoder model)
- Optional: LLM API keys for generation comparison

### download_corpus.sh
- `wget` or `curl`
- `jq` (for JSON parsing, optional)

### extract_pdf_metadata.py
- `pypdfium2` or `pdfplumber` (PDF text extraction)
- `json` (standard library)

### regenerate_corpus_from_metadata.py
- `json` (standard library)
- `os` and `hashlib` (file integrity checks)

## 📊 Script Statistics

- **Total Scripts**: 4
- **Diagnostic Tools**: 1
- **Download/Setup Scripts**: 1
- **Metadata Scripts**: 2

## ⚙️ Configuration

### Script Configuration Files
Most scripts use configuration from:
- `corpus/metadata.json` (document metadata)
- `src/config.py` (system configuration)
- Environment variables (API keys, paths)

### Logging
Scripts output to console with timestamps:
```
[2024-01-15 10:30:45] INFO: Starting corpus download...
[2024-01-15 10:30:50] INFO: Downloaded 1604.08127v1.pdf (628 KB)
```

## 🐛 Troubleshooting

### download_corpus.sh Fails
**Error:** `wget: command not found`

**Solution:** Install wget or modify script to use curl:
```bash
# Ubuntu/Debian
sudo apt-get install wget

# macOS
brew install wget
```

### extract_pdf_metadata.py Fails on PDF
**Error:** `PdfError: Could not extract text from PDF`

**Solution:**
1. Verify PDF is not corrupted
2. Try alternative PDF parser (pypdfium2 vs pdfplumber)
3. Check PDF encryption/permissions

### regenerate_corpus_from_metadata.py Reports Missing Files
**Error:** `File not found: corpus/2605.23950v1.pdf`

**Solution:**
1. Run download script to fetch missing files
2. Manually download from source URL in metadata.json
3. Remove missing file entries from metadata.json

## 📝 Adding New Scripts

When adding new utility scripts:
1. Follow naming convention (lowercase with underscores)
2. Add `--help` documentation
3. Include usage examples in script header
4. Update this README with script description
5. Add error handling and logging
6. Test on both Windows and Unix-like systems

## 🔗 Related Documentation

- **Corpus Management**: See [corpus/README.md](../corpus/README.md)
- **Ingestion Pipeline**: See [src/ingestion/README.md](../src/ingestion/README.md)
- **Retrieval Diagnostics**: See [tests/dump_full_snippets.py](../tests/dump_full_snippets.py)
- **Agent Guidelines**: See [AGENTS.md](../AGENTS.md)
