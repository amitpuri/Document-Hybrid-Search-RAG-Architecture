"""
Ingestion pipeline package: extractors, chunkers, caching, storage, and pipeline orchestrator.
"""

from src.ingestion.extractors import extract_text_from_pdf, iter_pdf_documents
from src.ingestion.chunkers import chunk_document_structured, split_into_sentences
from src.ingestion.cache import compute_cache_key, load_cached_chunks, save_cached_chunks
from src.ingestion.storage import BaseChunkStore, InMemoryChunkStore
from src.ingestion.pipeline import IngestionPipeline, parse_formatted_chunk_string

__all__ = [
    "extract_text_from_pdf",
    "iter_pdf_documents",
    "chunk_document_structured",
    "split_into_sentences",
    "compute_cache_key",
    "load_cached_chunks",
    "save_cached_chunks",
    "BaseChunkStore",
    "InMemoryChunkStore",
    "IngestionPipeline",
    "parse_formatted_chunk_string",
]
