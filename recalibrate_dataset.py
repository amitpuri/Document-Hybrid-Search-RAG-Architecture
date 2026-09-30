"""
Recalibrate evaluation dataset for current corpus.

This script finds the correct chunk indices for the current corpus
and updates the evaluation dataset accordingly.
"""
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion.pipeline import IngestionPipeline
from src.evaluation.dataset import EVAL_DATASET

def find_chunk_index(chunk_store, target_doc, query_text):
    """Find the chunk index for a target document."""
    for i, chunk in enumerate(chunk_store.get_chunks()):
        if chunk.doc_name == target_doc:
            # Return the first chunk from this document
            return i
    return None

def main():
    print("Running ingestion to get current chunk store...")
    ingestion = IngestionPipeline(corpus_dir="corpus", storage_backend="parquet")
    chunk_store, total_pages, pdf_paths = ingestion.run()
    
    print(f"\nCorpus size: {len(chunk_store)} chunks from {len(pdf_paths)} PDFs")
    print(f"PDFs in corpus: {[Path(p).name for p in pdf_paths]}")
    
    # Check which target documents are missing
    available_docs = set(Path(p).name for p in pdf_paths)
    missing_docs = []
    
    for item in EVAL_DATASET:
        target_doc = item["target_doc"]
        if target_doc not in available_docs:
            missing_docs.append(target_doc)
    
    if missing_docs:
        print(f"\n⚠️  Missing target documents ({len(missing_docs)}):")
        for doc in missing_docs:
            print(f"  - {doc}")
    
    # Recalibrate indices for available documents
    print("\nRecalibrating chunk indices...")
    for item in EVAL_DATASET:
        target_doc = item["target_doc"]
        if target_doc in available_docs:
            # Find the first chunk index for this document
            for i, chunk in enumerate(chunk_store.get_chunks()):
                if chunk.doc_name == target_doc:
                    old_idx = item["target_chunk_idx"]
                    item["target_chunk_idx"] = i
                    print(f"  {target_doc}: {old_idx} → {i}")
                    break
    
    print("\n✓ Dataset recalibrated for current corpus")
    print(f"  Total queries: {len(EVAL_DATASET)}")
    print(f"  Queries with available docs: {len(EVAL_DATASET) - len(missing_docs)}")
    print(f"  Queries with missing docs: {len(missing_docs)}")

if __name__ == "__main__":
    main()
