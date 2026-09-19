"""
Multi-backend PDF text extraction.
Supports pypdfium2, pdfplumber, and pypdf/PyPDF2 in fallback order.
"""

import os
from pathlib import Path
from typing import Generator, List, Tuple, Union


def extract_text_from_pdf(pdf_path: Union[str, Path]) -> List[Tuple[int, str]]:
    """
    Extracts text page-by-page from a PDF file.

    Returns:
        List of tuples: (page_number_1_indexed, raw_page_text)
    """
    pdf_path = str(pdf_path)
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

    # 3. pypdf / PyPDF2
    try:
        import pypdf

        reader = pypdf.PdfReader(pdf_path)
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    except ImportError:
        pass

    try:
        import PyPDF2

        reader = PyPDF2.PdfReader(pdf_path)
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    except ImportError:
        raise ImportError("Please install pypdfium2, pdfplumber, or pypdf to extract PDF text.")


def iter_pdf_documents(
    corpus_dir: Union[str, Path],
) -> Generator[Tuple[str, List[Tuple[int, str]]], None, None]:
    """
    Batch generator over all PDF documents in a directory.
    Yields (pdf_filename, list_of_pages) for scalable ingestion.
    """
    corpus_path = Path(corpus_dir)
    pdf_files = sorted([f for f in corpus_path.iterdir() if f.suffix.lower() == ".pdf"])
    for f in pdf_files:
        pages = extract_text_from_pdf(f)
        yield f.name, pages
