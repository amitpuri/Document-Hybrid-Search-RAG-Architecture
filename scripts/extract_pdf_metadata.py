"""
PDF Metadata Extraction Utility

Extracts metadata from PDF files in the corpus directory and generates a JSON file
with structure similar to EVAL_DATASET for document cataloging and analysis.

Supports:
- PDF metadata extraction using pypdfium2
- arXiv API integration for academic papers
- Text parsing fallback for title/author extraction
- Optional Gemini LLM enhancement for missing metadata fields
- Line-by-line JSONL output for memory efficiency with large corpora
"""

# Requirements: pypdfium2, arxiv (optional), google-genai (optional for LLM), python-dotenv

import sys
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import hashlib
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Load environment variables from .env file
load_dotenv()

try:
    import pypdfium2
except ImportError:
    print("Error: pypdfium2 is required. Install with: pip install pypdfium2")
    sys.exit(1)

try:
    import arxiv
    HAS_ARXIV = True
except ImportError:
    HAS_ARXIV = False

try:
    from google import genai
    HAS_GEMINI = True
except ImportError:
    try:
        import google.generativeai as genai
        HAS_GEMINI = True
    except ImportError:
        HAS_GEMINI = False
        print("Warning: google-genai not available. Install with: pip install google-genai for LLM-enhanced metadata")


def parse_arxiv_id_from_filename(filename: str) -> str:
    """
    Extract arXiv ID from filename (e.g., "1604.08127v1.pdf" -> "1604.08127v1").
    
    Args:
        filename: PDF filename
        
    Returns:
        arXiv ID without extension, or empty string if not found
    """
    if filename.endswith('.pdf'):
        filename = filename[:-4]
    return filename


def fetch_arxiv_metadata(arxiv_id: str) -> Dict[str, str]:
    """
    Fetch metadata from arXiv API for a given arXiv ID.
    
    Args:
        arxiv_id: arXiv identifier (e.g., "1604.08127v1")
        
    Returns:
        Dictionary with arXiv metadata (title, authors, abstract, etc.)
    """
    if not HAS_ARXIV:
        return {}
    
    try:
        # Extract base ID (remove version for search)
        base_id = arxiv_id.split('v')[0] if 'v' in arxiv_id else arxiv_id
        
        # Use the correct API method for current arxiv library version
        search = arxiv.Search(id_list=[base_id])
        client = arxiv.Client()
        results = client.results(search)
        
        for result in results:
            return {
                "title": result.title,
                "authors": ", ".join(author.name for author in result.authors),
                "abstract": result.summary,
                "published": result.published.strftime("%Y-%m-%d") if result.published else "",
                "arxiv_id": arxiv_id,
                "categories": ", ".join(result.categories) if result.categories else "",
            }
    except Exception as e:
        print(f"  Warning: Failed to fetch arXiv metadata for {arxiv_id}: {e}")
    
    return {}


def parse_title_from_first_page(text: str) -> str:
    """
    Attempt to extract title from first page text using common academic paper patterns.
    
    Args:
        text: First page text content
        
    Returns:
        Extracted title or empty string
    """
    if not text:
        return ""
    
    lines = text.split('\n')
    
    # Common patterns for academic paper titles
    # Look for the first non-empty line that's reasonably long and in title case
    for i, line in enumerate(lines):
        line = line.strip()
        if line and len(line) > 10 and len(line) < 200:
            # Skip common header/footer patterns
            skip_patterns = ['arXiv:', 'http://', 'https://', 'DOI:', 'doi:', 'Abstract', 'ABSTRACT']
            if not any(pattern in line.lower() for pattern in skip_patterns):
                # Check if it looks like a title (mixed case, not all caps or all lowercase)
                if line[0].isupper() and not line.isupper() and not line.islower():
                    return line
    
    return ""


def parse_authors_from_first_page(text: str) -> str:
    """
    Attempt to extract authors from first page text using common academic paper patterns.
    
    Args:
        text: First page text content
        
    Returns:
        Extracted authors string or empty string
    """
    if not text:
        return ""
    
    lines = text.split('\n')
    
    # Look for author patterns (often after title, before abstract)
    for i, line in enumerate(lines):
        line = line.strip()
        # Common author patterns: names separated by commas or "and"
        if line and len(line) < 200:
            # Check if line contains name-like patterns
            words = line.split()
            if len(words) >= 2 and len(words) <= 10:
                # Check if words look like names (capitalized, contains @ or affiliation markers)
                name_indicators = sum(1 for word in words if word[0].isupper() and len(word) > 1)
                if name_indicators >= len(words) * 0.5:
                    return line
    
    return ""


def init_gemini_client():
    """Initialize Gemini client using environment variables."""
    if not HAS_GEMINI:
        return None
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY not set in environment")
        return None
    
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except ImportError:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            return genai.GenerativeModel("gemini-3.8-flash")
        except Exception as e:
            print(f"Warning: Failed to initialize Gemini client: {e}")
            return None


def extract_metadata_with_gemini(first_page_text: str, full_preview: str = "") -> Dict[str, str]:
    """
    Use Gemini LLM to extract missing metadata from PDF content.
    
    Args:
        first_page_text: Text content from the first page
        full_preview: Additional text preview from the document
        
    Returns:
        Dictionary with extracted metadata fields
    """
    if not HAS_GEMINI:
        return {}
    
    client = init_gemini_client()
    if not client:
        return {}
    
    # Combine text for analysis
    combined_text = first_page_text
    if full_preview:
        combined_text += "\n\n" + full_preview
    
    # Limit text length to avoid token limits
    combined_text = combined_text[:8000]  # ~2000 tokens
    
    prompt = f"""Analyze the following academic paper text and extract metadata in JSON format:

Text:
{combined_text}

Please extract and return a JSON object with these fields:
- title: The paper title
- authors: List of authors (comma-separated)
- subject: Main subject area or topic
- keywords: Key terms/topics (comma-separated, max 5)

Return ONLY the JSON object, no other text."""

    try:
        if hasattr(client, "models"):
            from google.genai import types as genai_types
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=500,
                )
            )
        else:
            response = client.generate_content(prompt)
        
        # Extract JSON from response
        response_text = response.text if hasattr(response, 'text') else str(response)
        
        # Try to parse JSON from response
        import re
        json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
        if json_match:
            try:
                import json
                metadata = json.loads(json_match.group())
                return {
                    "title": metadata.get("title", ""),
                    "author": metadata.get("authors", ""),
                    "subject": metadata.get("subject", ""),
                    "keywords": metadata.get("keywords", ""),
                }
            except json.JSONDecodeError:
                pass
        
        return {}
        
    except Exception as e:
        print(f"  Warning: Gemini metadata extraction failed: {e}")
        return {}


def extract_pdf_metadata(pdf_path: Path, use_llm: bool = False) -> Dict[str, Any]:
    """
    Extract comprehensive metadata from a PDF file using pypdfium2,
    with arXiv API fallback, text parsing, and optional LLM enhancement.
    
    Args:
        pdf_path: Path to the PDF file
        use_llm: Whether to use Gemini LLM to fill missing metadata fields
        
    Returns:
        Dictionary containing extracted metadata
    """
    metadata = {
        "filename": pdf_path.name,
        "filepath": str(pdf_path),
        "file_size_bytes": pdf_path.stat().st_size,
        "file_size_mb": round(pdf_path.stat().st_size / (1024 * 1024), 2),
    }
    
    first_page_text = ""
    
    try:
        pdf = pypdfium2.PdfDocument(str(pdf_path))
        
        # Basic PDF metadata
        pdf_metadata = pdf.get_metadata_dict()
        
        metadata.update({
            "title": pdf_metadata.get("Title", "").strip(),
            "author": pdf_metadata.get("Author", "").strip(),
            "subject": pdf_metadata.get("Subject", "").strip(),
            "keywords": pdf_metadata.get("Keywords", "").strip(),
            "creator": pdf_metadata.get("Creator", "").strip(),
            "producer": pdf_metadata.get("Producer", "").strip(),
            "creation_date": pdf_metadata.get("CreationDate", "").strip(),
            "modification_date": pdf_metadata.get("ModDate", "").strip(),
        })
        
        # Page count
        metadata["page_count"] = len(pdf)
        
        # Extract first page text for preview/sample and parsing (optional, can fail)
        try:
            if len(pdf) > 0:
                first_page = pdf[0]
                text_page = first_page.get_textpage()
                first_page_text = text_page.get_text_range()
                # Clean up the text and take first 500 chars for preview
                text_preview = " ".join(first_page_text.split())[:500]
                metadata["first_page_preview"] = text_preview
        except:
            metadata["first_page_preview"] = ""
        
        # Extract last page text for preview (optional, can fail)
        try:
            if len(pdf) > 1:
                last_page = pdf[-1]
                text_page = last_page.get_textpage()
                text = text_page.get_text_range()
                text_preview = " ".join(text.split())[:500]
                metadata["last_page_preview"] = text_preview
        except:
            metadata["last_page_preview"] = ""
        
        pdf.close()
        
    except Exception as e:
        metadata["extraction_error"] = str(e)
        metadata["page_count"] = 0
        metadata["first_page_preview"] = ""
        metadata["last_page_preview"] = ""
        metadata.setdefault("title", "")
        metadata.setdefault("author", "")
        metadata.setdefault("subject", "")
        metadata.setdefault("keywords", "")
        metadata.setdefault("creator", "")
        metadata.setdefault("producer", "")
        metadata.setdefault("creation_date", "")
        metadata.setdefault("modification_date", "")
    
    # Fallback: Try arXiv API if filename matches arXiv pattern (optional, requires network)
    if HAS_ARXIV and (not metadata.get("title") or not metadata.get("author")):
        arxiv_id = parse_arxiv_id_from_filename(pdf_path.name)
        if arxiv_id and ('.' in arxiv_id or arxiv_id.startswith('arXiv')):
            arxiv_metadata = fetch_arxiv_metadata(arxiv_id)
            if arxiv_metadata:
                if not metadata.get("title"):
                    metadata["title"] = arxiv_metadata.get("title", "")
                if not metadata.get("author"):
                    metadata["author"] = arxiv_metadata.get("authors", "")
                metadata["arxiv_abstract"] = arxiv_metadata.get("abstract", "")
                metadata["arxiv_published"] = arxiv_metadata.get("published", "")
                metadata["arxiv_categories"] = arxiv_metadata.get("categories", "")
    
    # Fallback: Parse title and author from first page text
    if not metadata.get("title") and first_page_text:
        parsed_title = parse_title_from_first_page(first_page_text)
        if parsed_title:
            metadata["title"] = parsed_title
    
    if not metadata.get("author") and first_page_text:
        parsed_authors = parse_authors_from_first_page(first_page_text)
        if parsed_authors:
            metadata["author"] = parsed_authors
    
    # Optional: Use Gemini LLM to fill missing metadata fields
    if use_llm and (not metadata.get("title") or not metadata.get("author") or not metadata.get("subject")):
        print(f"  Using Gemini LLM to extract missing metadata for {pdf_path.name}...")
        llm_metadata = extract_metadata_with_gemini(first_page_text, metadata.get("first_page_preview", ""))
        if llm_metadata:
            if not metadata.get("title") and llm_metadata.get("title"):
                metadata["title"] = llm_metadata["title"]
            if not metadata.get("author") and llm_metadata.get("author"):
                metadata["author"] = llm_metadata["author"]
            if not metadata.get("subject") and llm_metadata.get("subject"):
                metadata["subject"] = llm_metadata["subject"]
            if not metadata.get("keywords") and llm_metadata.get("keywords"):
                metadata["keywords"] = llm_metadata["keywords"]
            metadata["llm_enhanced"] = True
    
    # Generate file hash for uniqueness
    metadata["sha256_hash"] = compute_file_hash(pdf_path)
    
    return metadata


def compute_file_hash(file_path: Path) -> str:
    """
    Compute SHA-256 hash of a file for integrity checking.
    
    Args:
        file_path: Path to the file
        
    Returns:
        SHA-256 hash as hexadecimal string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def extract_corpus_metadata(corpus_dir: Path, use_llm: bool = False, output_path: Path = None, line_by_line: bool = False) -> List[Dict[str, Any]]:
    """
    Extract metadata from all PDF files in the corpus directory.
    
    Args:
        corpus_dir: Path to the corpus directory
        use_llm: Whether to use Gemini LLM to fill missing metadata fields
        output_path: Path to output file for line-by-line saving
        line_by_line: Whether to save entries line by line (JSONL format)
        
    Returns:
        List of metadata dictionaries for each PDF
    """
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
    
    pdf_files = sorted(corpus_dir.glob("*.pdf"))
    
    if not pdf_files:
        raise ValueError(f"No PDF files found in: {corpus_dir}")
    
    print(f"Found {len(pdf_files)} PDF files in {corpus_dir}")
    print("Extracting metadata...")
    if use_llm:
        print("LLM enhancement enabled for missing metadata fields")
    if line_by_line:
        print("Line-by-line saving enabled (JSONL format)")
    
    metadata_list = []
    
    # If line-by-line saving, open file and write incrementally
    if line_by_line and output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, pdf_path in enumerate(pdf_files, 1):
                print(f"  [{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
                metadata = extract_pdf_metadata(pdf_path, use_llm=use_llm)
                metadata_list.append(metadata)
                # Write immediately to file
                f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
                f.flush()  # Ensure immediate write
    else:
        # Standard batch processing
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"  [{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
            metadata = extract_pdf_metadata(pdf_path, use_llm=use_llm)
            metadata_list.append(metadata)
    
    return metadata_list


def generate_dataset_structure(metadata_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert raw metadata into a structure similar to EVAL_DATASET.
    
    This creates a catalog format that can be used for:
    - Document discovery and browsing
    - Query generation for evaluation
    - Corpus analysis and statistics
    
    Args:
        metadata_list: List of raw metadata dictionaries
        
    Returns:
        List of dataset entries with EVAL_DATASET-like structure
    """
    dataset = []
    
    for metadata in metadata_list:
        entry = {
            "filename": metadata["filename"],
            "title": metadata["title"] or metadata["filename"],
            "author": metadata["author"] or "Unknown",
            "subject": metadata["subject"] or "",
            "keywords": metadata["keywords"] or "",
            "page_count": metadata["page_count"],
            "file_size_mb": metadata["file_size_mb"],
            "sha256_hash": metadata["sha256_hash"],
            "first_page_preview": metadata.get("first_page_preview", ""),
            "last_page_preview": metadata.get("last_page_preview", ""),
            # arXiv-specific fields if available
            "arxiv_abstract": metadata.get("arxiv_abstract", ""),
            "arxiv_published": metadata.get("arxiv_published", ""),
            "arxiv_categories": metadata.get("arxiv_categories", ""),
            # Placeholder fields for potential query generation
            "suggested_queries": [],
            "target_entities": [],
            "target_relations": [],
            "is_multihop": False,
        }
        dataset.append(entry)
    
    return dataset


def save_metadata_json(metadata_list: List[Dict[str, Any]], output_path: Path, line_by_line: bool = False) -> None:
    """
    Save metadata list to a JSON file.
    
    Args:
        metadata_list: List of metadata dictionaries
        output_path: Path to output JSON file
        line_by_line: If True, write JSON entries line by line (JSONL format)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if line_by_line:
        # Save as JSONL (one JSON object per line)
        with open(output_path, 'w', encoding='utf-8') as f:
            for metadata in metadata_list:
                f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
        print(f"\nMetadata saved to: {output_path} (JSONL format)")
    else:
        # Save as standard JSON array
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, indent=2, ensure_ascii=False)
        print(f"\nMetadata saved to: {output_path}")


def print_summary(metadata_list: List[Dict[str, Any]]) -> None:
    """
    Print summary statistics of the extracted metadata.
    
    Args:
        metadata_list: List of metadata dictionaries
    """
    total_files = len(metadata_list)
    total_pages = sum(m.get("page_count", 0) for m in metadata_list)
    total_size_mb = sum(m.get("file_size_mb", 0) for m in metadata_list)
    
    with_title = sum(1 for m in metadata_list if m.get("title"))
    with_author = sum(1 for m in metadata_list if m.get("author"))
    with_subject = sum(1 for m in metadata_list if m.get("subject"))
    with_errors = sum(1 for m in metadata_list if m.get("extraction_error"))
    with_llm_enhanced = sum(1 for m in metadata_list if m.get("llm_enhanced"))
    
    print("\n" + "=" * 60)
    print("CORPUS METADATA SUMMARY")
    print("=" * 60)
    print(f"Total PDF files: {total_files}")
    print(f"Total pages: {total_pages}")
    print(f"Total size: {total_size_mb:.2f} MB")
    if total_files > 0:
        print(f"Average pages per file: {total_pages / total_files:.1f}")
        print(f"Average size per file: {total_size_mb / total_files:.2f} MB")
    print(f"\nMetadata coverage:")
    print(f"  Files with title: {with_title}/{total_files} ({100*with_title/total_files:.1f}%)")
    print(f"  Files with author: {with_author}/{total_files} ({100*with_author/total_files:.1f}%)")
    print(f"  Files with subject: {with_subject}/{total_files} ({100*with_subject/total_files:.1f}%)")
    if with_errors > 0:
        print(f"  Files with extraction errors: {with_errors}/{total_files} ({100*with_errors/total_files:.1f}%)")
    if with_llm_enhanced > 0:
        print(f"  Files with LLM-enhanced metadata: {with_llm_enhanced}/{total_files} ({100*with_llm_enhanced/total_files:.1f}%)")
    print("=" * 60)


def main():
    """Main entry point for the metadata extraction utility."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract metadata from PDF files in the corpus directory"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="corpus",
        help="Path to corpus directory (default: corpus)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="corpus_metadata.json",
        help="Output JSON file path (default: corpus_metadata.json)"
    )
    parser.add_argument(
        "--dataset-format",
        action="store_true",
        help="Output in EVAL_DATASET-like structure instead of raw metadata"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use Gemini LLM to fill missing metadata fields (requires GEMINI_API_KEY)"
    )
    parser.add_argument(
        "--line-by-line",
        action="store_true",
        help="Save entries line by line (JSONL format) for memory efficiency (incompatible with --dataset-format)"
    )
    
    args = parser.parse_args()
    
    # Check for incompatible options
    if args.line_by_line and args.dataset_format:
        print("Error: --line-by-line and --dataset-format are incompatible options", file=sys.stderr)
        sys.exit(1)
    
    corpus_dir = Path(args.corpus)
    output_path = Path(args.output)
    
    try:
        # Extract metadata from all PDFs
        if args.line_by_line:
            # Line-by-line mode: save incrementally during extraction
            metadata_list = extract_corpus_metadata(
                corpus_dir, 
                use_llm=args.use_llm,
                output_path=output_path,
                line_by_line=True
            )
            print(f"\nLine-by-line JSONL saved to: {output_path}")
        else:
            # Standard mode: extract all, then save
            metadata_list = extract_corpus_metadata(corpus_dir, use_llm=args.use_llm)
        
        # Choose output format (only if not already saved line-by-line)
        if not args.line_by_line:
            if args.dataset_format:
                output_data = generate_dataset_structure(metadata_list)
                print("\nGenerating EVAL_DATASET-like structure...")
            else:
                output_data = metadata_list
            
            # Save to JSON
            save_metadata_json(output_data, output_path, line_by_line=args.line_by_line)
        
        # Print summary
        print_summary(metadata_list)
        
        print(f"\n✓ Successfully extracted metadata for {len(metadata_list)} PDF files")
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
