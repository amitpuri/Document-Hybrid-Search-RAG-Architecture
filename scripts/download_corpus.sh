#!/bin/bash

# Download arXiv papers and ingest them into the corpus.
#
# Usage:
#   ./scripts/download_corpus.sh <arxiv_id> [<arxiv_id> ...]
#   ./scripts/download_corpus.sh --batch corpus/MANIFEST.md
#   ./scripts/download_corpus.sh --category cs.AI --max-results 50

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
CORPUS_DIR="${REPO_ROOT}/corpus"

# Default options
CATEGORY="cs.AI"
MAX_RESULTS=20
BATCH_FILE=""
ARXIV_IDS=()
STORAGE_BACKEND="parquet"
BATCH_SIZE=10

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch)
            BATCH_FILE="$2"
            shift 2
            ;;
        --category)
            CATEGORY="$2"
            shift 2
            ;;
        --max-results)
            MAX_RESULTS="$2"
            shift 2
            ;;
        --backend)
            STORAGE_BACKEND="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --corpus-dir)
            CORPUS_DIR="$2"
            shift 2
            ;;
        --help|-h)
            cat <<EOF
Usage: $(basename "$0") [OPTIONS] [<arxiv_id> ...]

Download and ingest arXiv papers into the corpus.

Options:
  --batch FILE              Load arxiv_ids from MANIFEST.md file
  --category CAT            arXiv category (default: cs.AI)
  --max-results N           Max results per category search (default: 20)
  --backend BACKEND         Storage backend: parquet, memory, qdrant (default: parquet)
  --batch-size N            Papers per ingestion batch (default: 10)
  --corpus-dir DIR          Corpus directory (default: ./corpus)
  --help                    Show this help message

Examples:
  # Download specific papers
  ./scripts/download_corpus.sh 2301.07041 2301.07042

  # Download from batch manifest
  ./scripts/download_corpus.sh --batch corpus/MANIFEST.md

  # Search and download from category
  ./scripts/download_corpus.sh --category cs.IR --max-results 50
EOF
            exit 0
            ;;
        *)
            # Assume it's an arxiv_id
            ARXIV_IDS+=("$1")
            shift
            ;;
    esac
done

# Ensure corpus directory exists
mkdir -p "$CORPUS_DIR"

# Run ingestion via Python
python3 << 'PYTHON_SCRIPT'
import sys
import json
from pathlib import Path

# Add repo to path
sys.path.insert(0, """$REPO_ROOT""")

from src.mcp.arxiv_server import ingest_to_corpus

arxiv_ids = """$ARXIV_IDS""".split()
corpus_dir = """$CORPUS_DIR"""
backend = """$STORAGE_BACKEND"""
batch_size = """$BATCH_SIZE"""

if not arxiv_ids:
    print("Error: No arXiv IDs provided. Use --help for usage.")
    sys.exit(1)

print(f"Ingesting {len(arxiv_ids)} papers into {corpus_dir}...")
result = ingest_to_corpus(
    arxiv_ids=arxiv_ids,
    corpus_dir=corpus_dir,
    storage_backend=backend,
    batch_size=int(batch_size)
)

print(json.dumps(result, indent=2))

if result["failed"]:
    print(f"\nWarning: {len(result['failed'])} papers failed to ingest:", file=sys.stderr)
    for arxiv_id in result["failed"]:
        print(f"  - {arxiv_id}", file=sys.stderr)
    sys.exit(1)

print(f"\nSuccessfully ingested {result['ingested']} papers.")
PYTHON_SCRIPT

exit_code=$?
exit "$exit_code"
