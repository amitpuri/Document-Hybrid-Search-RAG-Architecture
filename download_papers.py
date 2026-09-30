"""
Download specific arXiv papers needed for validation.
"""
import sys
from pathlib import Path
from urllib.request import urlretrieve

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Papers needed for validation
PAPERS = [
    "2605.23950v1",
    "2605.10223v1", 
    "2604.00073v3",
    "2604.22750v2",
    "2608.18613v1",
    "2609.09153v1",
]

def main():
    corpus_dir = Path("corpus")
    corpus_dir.mkdir(exist_ok=True)
    
    for arxiv_id in PAPERS:
        pdf_filename = f"{arxiv_id}.pdf"
        pdf_path = corpus_dir / pdf_filename
        
        if pdf_path.exists():
            print(f"✓ {pdf_filename} already exists, skipping")
            continue
        
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        print(f"Downloading {pdf_filename} from {pdf_url}...")
        
        try:
            urlretrieve(pdf_url, str(pdf_path))
            print(f"✓ Downloaded {pdf_filename}")
        except Exception as e:
            print(f"✗ Failed to download {pdf_filename}: {e}")
    
    print("\nDownload complete!")
    print(f"Corpus directory: {corpus_dir}")
    print(f"PDFs downloaded: {len([p for p in PAPERS if (corpus_dir / f'{p}.pdf').exists()])}/{len(PAPERS)}")

if __name__ == "__main__":
    main()
