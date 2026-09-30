"""
Corpus Regeneration Script

This script can be used to regenerate the corpus from arXiv IDs listed in corpus/metadata.json
instead of vendoring the PDFs directly in the repository. This addresses the concern about
redistributing third-party PDFs verbatim.

Usage:
    python scripts/regenerate_corpus_from_metadata.py --corpus-dir corpus
"""

import sys
import argparse
import json
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description="Regenerate corpus from arXiv metadata")
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default="corpus",
        help="Directory containing metadata.json and where PDFs will be downloaded"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print arXiv IDs to download without actually downloading"
    )
    
    args = parser.parse_args()
    
    corpus_dir = Path(args.corpus_dir)
    metadata_file = corpus_dir / "metadata.json"
    
    if not metadata_file.exists():
        print(f"Error: {metadata_file} not found")
        return 1
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    arxiv_ids = []
    for item in metadata:
        filename = item.get("filename", "")
        # Extract arXiv ID from filename (e.g., "1604.08127v1.pdf" -> "1604.08127v1")
        if filename.endswith(".pdf"):
            arxiv_id = filename[:-4]  # Remove .pdf
            arxiv_ids.append(arxiv_id)
    
    print(f"Found {len(arxiv_ids)} papers in metadata.json")
    print(f"arXiv IDs: {', '.join(arxiv_ids[:5])}{'...' if len(arxiv_ids) > 5 else ''}")
    
    if args.dry_run:
        print("\nDry run mode - no downloads performed")
        print("To actually download, run without --dry-run flag")
        print("\nThen use the arXiv ingestion pipeline:")
        print("python -m src.cli ingest --arxiv --arxiv-ids <comma-separated-ids>")
        return 0
    
    print("\nTo regenerate the corpus, use the arXiv ingestion pipeline:")
    print(f"python -m src.cli ingest --arxiv --arxiv-ids {','.join(arxiv_ids)}")
    print("\nOr use category-based ingestion:")
    print("python -m src.cli ingest --arxiv --category cs.AI,cs.IR,cs.CL --limit 50")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())