"""
Consolidated Corpus Loading and Extraction Module.

Provides:
- extract_text_from_pdf: Multi-backend PDF text extraction (pypdfium2 -> pdfplumber -> PyPDF2)
- load_pdf_corpus: Fast sliding-window word-level chunking
- load_structured_corpus: Sentence-aware and section-header-preserving chunking
- Parameter- and mtime-sensitive disk caching (SHA-256 cache keys)
- Shared tokenization and stopword utilities
"""

import os
import re
import sys
import pickle
import hashlib

# Reconfigure stdout for UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ===========================================================
# Standard Tokenization & Stopwords
# ===========================================================
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
    "by", "of", "from", "as", "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "this", "that", "these", "those",
    "it", "its", "we", "our", "you", "your", "they", "their", "he", "she", "which",
    "can", "will", "would", "should", "could", "all", "any", "both", "each", "more"
}

def tokenize(text, remove_stopwords=False):
    """Lowercases and cleans text into whitespace-delimited word tokens."""
    cleaned = "".join(ch if ch.isalnum() else " " for ch in text.lower())
    tokens = cleaned.split()
    if remove_stopwords:
        return [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


# ===========================================================
# PDF Extraction Backend
# ===========================================================
def extract_text_from_pdf(pdf_path):
    """
    Extracts text page-by-page using pypdfium2, pdfplumber, or PyPDF2.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # 1. pypdfium2 (fastest, clean text ranges)
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

    # 2. pdfplumber
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

    # 3. PyPDF2
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(pdf_path)
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    except ImportError:
        raise ImportError("Please install pypdfium2, pdfplumber, or PyPDF2 to read PDFs.")


# ===========================================================
# Standard Sliding-Window Chunking (Word Count Based)
# ===========================================================
def load_pdf_corpus(pdf_inputs, chunk_size=120, overlap=30):
    """
    Loads one or more PDFs and chunks content via word-based sliding windows.
    """
    if isinstance(pdf_inputs, (str, os.PathLike)):
        pdf_inputs = [pdf_inputs]

    chunks = []
    total_pages = 0
    multi_doc = len(pdf_inputs) > 1

    for pdf_path in pdf_inputs:
        if not os.path.exists(pdf_path):
            continue
        doc_name = os.path.basename(pdf_path)
        pages = extract_text_from_pdf(pdf_path)
        total_pages += len(pages)
        for page_num, text in pages:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            page_clean = " ".join(lines)
            words = page_clean.split()
            if not words:
                continue
            step = max(1, chunk_size - overlap)
            for i in range(0, len(words), step):
                chunk_words = words[i : i + chunk_size]
                if len(chunk_words) >= 25:
                    prefix = f"[{doc_name} | Page {page_num}] " if multi_doc else f"[Page {page_num}] "
                    chunks.append(prefix + " ".join(chunk_words))

    return chunks, total_pages


# ===========================================================
# Structure- & Sentence-Aware Chunking
# ===========================================================
SECTION_HEADER_REGEX = re.compile(
    r"^(?:(?:\d+\.?)+\s+[A-Z][A-Za-z0-9\s—–:-]{2,60}|(?:Abstract|Introduction|Conclusion|References|Related Work|Methodology|Discussion|Background))\s*$",
    re.MULTILINE
)

def split_into_sentences(text):
    """Splits raw text into natural sentences using punctuation boundaries."""
    paragraphs = text.split("\n\n")
    sentences = []
    for para in paragraphs:
        cleaned_para = " ".join(line.strip() for line in para.splitlines() if line.strip())
        if not cleaned_para:
            continue
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
                chunk_text = " ".join(buffer_sents)
                prefix = f"[{doc_name} | Page {page_num} | § {current_section}] "
                chunks.append(prefix + chunk_text)

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
# Parameter- & State-Sensitive Disk Cache
# ===========================================================
def compute_cache_key(corpus_dir, max_words=120, overlap_sentences=1, extra_tag=""):
    """
    Computes a cryptographic hash over corpus files (name, mtime, size), chunking params,
    and any extra identifier tag (e.g. model name).
    Ensures that modifying files or changing parameters invalidates stale cache automatically.
    """
    hasher = hashlib.sha256()
    hasher.update(str(os.path.abspath(corpus_dir)).encode("utf-8"))
    hasher.update(f"max_words:{max_words},overlap_sents:{overlap_sentences},extra:{extra_tag}".encode("utf-8"))

    pdf_files = sorted([f for f in os.listdir(corpus_dir) if f.lower().endswith(".pdf")])
    for f in pdf_files:
        full_path = os.path.join(corpus_dir, f)
        try:
            stat = os.stat(full_path)
            hasher.update(f"{f}:{stat.st_mtime}:{stat.st_size}".encode("utf-8"))
        except OSError:
            pass

    return hasher.hexdigest()[:16]

_compute_cache_key = compute_cache_key


def load_structured_corpus(corpus_dir, cache_dir=".cache", force_rebuild=False, max_words=120, overlap_sentences=1):
    """
    Loads all PDFs in corpus_dir using structured chunking.
    Uses SHA-256 cache keying for safe, automatic invalidation.
    """
    cache_key = compute_cache_key(corpus_dir, max_words, overlap_sentences)
    cache_path = os.path.join(cache_dir, f"structured_cache_{cache_key}.pkl")

    if not force_rebuild and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
                return cached_data["chunks"], cached_data["total_pages"], cached_data["pdf_paths"]
        except Exception:
            pass  # Fallback to rebuild if read errors occur

    pdf_paths = [
        os.path.join(corpus_dir, f)
        for f in sorted(os.listdir(corpus_dir))
        if f.lower().endswith(".pdf")
    ]

    all_chunks = []
    total_pages = 0
    for path in pdf_paths:
        chunks, pages = chunk_document_structured(path, max_words=max_words, overlap_sentences=overlap_sentences)
        all_chunks.extend(chunks)
        total_pages += pages

    # Save cache
    os.makedirs(cache_dir, exist_ok=True)
    try:
        with open(cache_path, "wb") as f:
            pickle.dump({
                "chunks": all_chunks,
                "total_pages": total_pages,
                "pdf_paths": pdf_paths,
            }, f)
    except Exception:
        pass

    return all_chunks, total_pages, pdf_paths
