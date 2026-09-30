"""
Download all arXiv papers from metadata.json to match expected corpus size.
"""
import sys
import json
from pathlib import Path
from urllib.request import urlretrieve

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def main():
    corpus_dir = Path("corpus")
    corpus_dir.mkdir(exist_ok=True)
    
    # Load metadata
    with open("corpus/metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    
    # Extract arXiv IDs
    arxiv_ids = [item["filename"][:-4] for item in metadata if item["filename"].endswith(".pdf")]
    
    print(f"Found {len(arxiv_ids)} papers in metadata.json")
    print(f"Downloading to: {corpus_dir}\n")
    
    downloaded = 0
    skipped = 0
    failed = 0
    
    for i, arxiv_id in enumerate(arxiv_ids):
        pdf_filename = f"{arxiv_id}.pdf"
        pdf_path = corpus_dir / pdf_filename
        
        if pdf_path.exists():
            print(f"[{i+1}/{len(arxiv_ids)}] ✓ {pdf_filename} already exists, skipping")
            skipped += 1
            continue
        
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        print(f"[{i+1}/{len(arxiv_ids)}] Downloading {pdf_filename}...")
        
        try:
            urlretrieve(pdf_url, str(pdf_path))
            print(f"  ✓ Downloaded {pdf_filename}")
            downloaded += 1
        except Exception as e:
            print(f"  ✗ Failed to download {pdf_filename}: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"Total papers: {len(arxiv_ids)}")
    print(f"Downloaded: {downloaded}")
    print(f"Skipped (already exists): {skipped}")
    print(f"Failed: {failed}")
    print(f"Total PDFs in corpus: {len(list(corpus_dir.glob('*.pdf')))}")

if __name__ == "__main__":
    main()
