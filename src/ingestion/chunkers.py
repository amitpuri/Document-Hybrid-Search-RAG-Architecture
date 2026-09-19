"""
Structured, sentence-aware, and section-header-preserving document chunker.
"""

import re
from typing import List, Tuple

from src.common.types import DocumentChunk
from src.config import (
    DEFAULT_MAX_WORDS,
    DEFAULT_OVERLAP_SENTENCES,
    MIN_CHUNK_WORDS,
)

SECTION_HEADER_REGEX = re.compile(
    r"^(?:(?:\d+\.?)+\s+[A-Z][A-Za-z0-9\s—–:-]{2,60}|"
    r"(?:Abstract|Introduction|Conclusion|References|Related Work|"
    r"Methodology|Discussion|Background))\s*$",
    re.MULTILINE,
)


def split_into_sentences(text: str) -> List[str]:
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


def chunk_document_structured(
    doc_name: str,
    pages: List[Tuple[int, str]],
    start_chunk_id: int = 0,
    max_words: int = DEFAULT_MAX_WORDS,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
    min_chunk_words: int = MIN_CHUNK_WORDS,
) -> List[DocumentChunk]:
    """
    Chunks extracted pages into structured DocumentChunk records respecting
    sentence boundaries and tracking active section headers.
    """
    chunks: List[DocumentChunk] = []
    current_section = "Overview"
    current_id = start_chunk_id

    for page_num, raw_page_text in pages:
        lines = [line.strip() for line in raw_page_text.splitlines() if line.strip()]
        for line in lines:
            if SECTION_HEADER_REGEX.match(line) and len(line.split()) <= 8:
                current_section = line
                break

        sentences = split_into_sentences(raw_page_text)
        if not sentences:
            continue

        buffer_sents: List[str] = []
        buffer_words = 0

        for sent in sentences:
            sent_word_count = len(sent.split())
            if buffer_words + sent_word_count > max_words and buffer_sents:
                chunk_text = " ".join(buffer_sents)
                chunk = DocumentChunk(
                    chunk_id=current_id,
                    doc_name=doc_name,
                    page_num=page_num,
                    section=current_section,
                    text=chunk_text,
                )
                chunks.append(chunk)
                current_id += 1

                buffer_sents = buffer_sents[-overlap_sentences:] if overlap_sentences > 0 else []
                buffer_words = sum(len(s.split()) for s in buffer_sents)

            buffer_sents.append(sent)
            buffer_words += sent_word_count

        if buffer_sents and buffer_words >= min_chunk_words:
            chunk_text = " ".join(buffer_sents)
            chunk = DocumentChunk(
                chunk_id=current_id,
                doc_name=doc_name,
                page_num=page_num,
                section=current_section,
                text=chunk_text,
            )
            chunks.append(chunk)
            current_id += 1

    return chunks
