"""
Ingestion Pipeline orchestrator.
Handles PDF discovery, batch extraction, structured chunking, caching, and storage.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Optional
from src.common.types import DocumentChunk
from src.ingestion.extractors import extract_text_from_pdf
from src.ingestion.chunkers import chunk_document_structured
from src.ingestion.cache import (
    compute_cache_key,
    load_cached_chunks,
    save_cached_chunks,
)
from src.ingestion.storage import InMemoryChunkStore, BaseChunkStore
from src.config import CORPUS_DIR, CACHE_DIR, DEFAULT_MAX_WORDS, DEFAULT_OVERLAP_SENTENCES


_CHUNK_PREFIX_REGEX = re.compile(r"^\[(?P<doc>[^|]+)\s*\|\s*Page\s*(?P<page>\d+)\s*\|\s*§\s*(?P<section>[^\]]+)\]\s*(?P<text>.*)$", re.DOTALL)


def parse_formatted_chunk_string(formatted_str: str, chunk_id: int) -> DocumentChunk:
    """Parses a legacy or serialized formatted chunk string back into a DocumentChunk."""
    match = _CHUNK_PREFIX_REGEX.match(formatted_str.strip())
    if match:
        return DocumentChunk(
            chunk_id=chunk_id,
            doc_name=match.group("doc").strip(),
            page_num=int(match.group("page").strip()),
            section=match.group("section").strip(),
            text=match.group("text").strip(),
        )
    return DocumentChunk(
        chunk_id=chunk_id,
        doc_name="Unknown",
        page_num=1,
        section="Overview",
        text=formatted_str,
    )


class IngestionPipeline:
    """
    Orchestrates the document ingestion pipeline:
    1. Discovers PDF files
    2. Checks cryptographic disk cache
    3. Extracts text page-by-page
    4. Segments text into sentence-aware, section-preserving chunks
    5. Populates and returns a chunk store
    """

    def __init__(
        self,
        corpus_dir: str | Path = CORPUS_DIR,
        cache_dir: str | Path = CACHE_DIR,
        max_words: int = DEFAULT_MAX_WORDS,
        overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
    ):
        self.corpus_dir = Path(corpus_dir)
        self.cache_dir = Path(cache_dir)
        self.max_words = max_words
        self.overlap_sentences = overlap_sentences

    def run(self, force_rebuild: bool = False) -> Tuple[BaseChunkStore, int, List[str]]:
        """
        Executes the ingestion pipeline.

        Returns:
            Tuple: (chunk_store, total_pages, list_of_pdf_paths)
        """
        cache_key = compute_cache_key(
            self.corpus_dir, self.max_words, self.overlap_sentences
        )
        cache_file = self.cache_dir / f"structured_cache_{cache_key}.pkl"

        if not force_rebuild and cache_file.exists():
            cached_data = load_cached_chunks(cache_file)
            if cached_data is not None:
                raw_chunks = cached_data.get("chunks", [])
                total_pages = cached_data.get("total_pages", 0)
                pdf_paths = cached_data.get("pdf_paths", [])

                # Convert strings to DocumentChunk if needed
                chunks: List[DocumentChunk] = []
                for idx, item in enumerate(raw_chunks):
                    if isinstance(item, DocumentChunk):
                        chunks.append(item)
                    else:
                        chunks.append(parse_formatted_chunk_string(str(item), idx))

                store = InMemoryChunkStore(chunks)
                return store, total_pages, pdf_paths

        # Rebuild ingestion
        pdf_paths = sorted([
            str(p) for p in self.corpus_dir.glob("*.pdf")
        ])

        chunks: List[DocumentChunk] = []
        total_pages = 0
        current_id = 0

        for path in pdf_paths:
            doc_name = os.path.basename(path)
            pages = extract_text_from_pdf(path)
            total_pages += len(pages)
            doc_chunks = chunk_document_structured(
                doc_name=doc_name,
                pages=pages,
                start_chunk_id=current_id,
                max_words=self.max_words,
                overlap_sentences=self.overlap_sentences,
            )
            chunks.extend(doc_chunks)
            current_id += len(doc_chunks)

        store = InMemoryChunkStore(chunks)

        # Cache formatted text strings for compatibility
        save_data = {
            "chunks": store.get_texts(),
            "total_pages": total_pages,
            "pdf_paths": pdf_paths,
        }
        save_cached_chunks(cache_file, save_data)

        return store, total_pages, pdf_paths
