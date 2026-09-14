"""
Structure-Aware and Sentence-Aware PDF Chunking with Disk Caching.

Key Features:
1. Sentence boundary detection (avoids splitting mid-sentence or mid-citation).
2. Hierarchical Section Header tracking (prepends active section titles to descendant chunks).
3. Disk caching (stores extracted chunks in .cache/ to avoid re-extracting 354 pages on every run).
"""

import os
import re
import sys
import pickle
from pathlib import Path

# Reconfigure stdout for UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ===========================================================
# PDF Extraction
# ===========================================================
def extract_text_from_pdf(pdf_path):
    """
    Extracts text page-by-page using pypdfium2, pdfplumber, or PyPDF2.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(pdf_path)
        pages = []
        for i, page in enumerate(pdf):
            text = page.get_textpage().get_text_range()
            text = text.replace("\ufffe", "").replace("\xad", "")
            pages.append((i + 1, text))
        return pages
    except ImportError:
        pass

    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages.append((i + 1, text))
        return pages
    except ImportError:
        pass

    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(pdf_path)
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    except ImportError:
        raise ImportError("Please install pypdfium2, pdfplumber, or PyPDF2 to read PDFs.")


# ===========================================================
# Structure-Aware Sentence Splitting & Header Tracking
# ===========================================================
SECTION_HEADER_REGEX = re.compile(
    r"^(?:(?:\d+\.?)+\s+[A-Z][A-Za-z0-9\s—–:-]{2,60}|(?:Abstract|Introduction|Conclusion|References|Related Work|Methodology|Discussion|Background))\s*$",
    re.MULTILINE
)

def split_into_sentences(text):
    """Splits raw text into natural sentences using punctuation boundaries."""
    # Normalize line breaks within paragraphs
    paragraphs = text.split("\n\n")
    sentences = []
    for para in paragraphs:
        cleaned_para = " ".join(line.strip() for line in para.splitlines() if line.strip())
        if not cleaned_para:
            continue
        # Split on sentence terminals (. ? !) followed by space and capital letter
        raw_sents = re.split(r'(?<=[.?!])\s+(?=[A-Z0-9"“])', cleaned_para)
        for s in raw_sents:
            s_clean = s.strip()
            if s_clean:
                sentences.append(s_clean)
    return sentences


def chunk_document_structured(pdf_path, max_words=120, overlap_sentences=1):
    """
    Chunks a PDF document respecting sentence boundaries and maintaining active section headers.
    """
    doc_name = os.path.basename(pdf_path)
    pages = extract_text_from_pdf(pdf_path)

    chunks = []
    current_section = "Overview"

    for page_num, raw_page_text in pages:
        # Detect any section headers on this page
        lines = [l.strip() for l in raw_page_text.splitlines() if l.strip()]
        for line in lines:
            if SECTION_HEADER_REGEX.match(line) and len(line.split()) <= 8:
                current_section = line
                break

        sentences = split_into_sentences(raw_page_text)
        if not sentences:
            continue

        buffer_sents = []
        buffer_words = 0

        for sent in sentences:
            sent_word_count = len(sent.split())
            if buffer_words + sent_word_count > max_words and buffer_sents:
                # Flush chunk
                chunk_text = " ".join(buffer_sents)
                prefix = f"[{doc_name} | Page {page_num} | § {current_section}] "
                chunks.append(prefix + chunk_text)

                # Keep overlap sentences
                buffer_sents = buffer_sents[-overlap_sentences:] if overlap_sentences > 0 else []
                buffer_words = sum(len(s.split()) for s in buffer_sents)

            buffer_sents.append(sent)
            buffer_words += sent_word_count

        if buffer_sents and buffer_words >= 25:
            chunk_text = " ".join(buffer_sents)
            prefix = f"[{doc_name} | Page {page_num} | § {current_section}] "
            chunks.append(prefix + chunk_text)

    return chunks, len(pages)


# ===========================================================
# Corpus Loading with Disk Cache
# ===========================================================
def load_structured_corpus(corpus_dir, cache_dir=".cache", force_rebuild=False, max_words=120):
    """
    Loads all PDFs in corpus_dir using structured chunking.
    Caches parsed chunks to disk to enable instant reloading.
    """
    cache_path = os.path.join(cache_dir, "structured_corpus_cache.pkl")

    if not force_rebuild and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
                return cached_data["chunks"], cached_data["total_pages"], cached_data["pdf_paths"]
        except Exception:
            pass  # Rebuild if corrupted

    pdf_paths = [
        os.path.join(corpus_dir, f)
        for f in sorted(os.listdir(corpus_dir))
        if f.lower().endswith(".pdf")
    ]

    all_chunks = []
    total_pages = 0
    for path in pdf_paths:
        chunks, pages = chunk_document_structured(path, max_words=max_words)
        all_chunks.extend(chunks)
        total_pages += pages

    # Save to disk cache
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({
            "chunks": all_chunks,
            "total_pages": total_pages,
            "pdf_paths": pdf_paths,
        }, f)

    return all_chunks, total_pages, pdf_paths


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    corpus_folder = os.path.join(base_dir, "corpus")
    cache_folder = os.path.join(base_dir, ".cache")

    print(f"Loading and chunking corpus from: {corpus_folder}")
    chunks, pages, pdfs = load_structured_corpus(corpus_folder, cache_dir=cache_folder, force_rebuild=True)
    print(f"Parsed {len(pdfs)} PDFs across {pages} pages into {len(chunks)} structure-aware chunks.")
    print("\nSample Chunk with Header Provenance:")
    print("-" * 80)
    print(chunks[0])
    print("-" * 80)
    print(chunks[10])
