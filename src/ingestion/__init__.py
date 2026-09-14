"""
Ingestion pipeline package: extractors, chunkers, caching, storage, and pipeline orchestrator.
"""

from src.ingestion.extractors import extract_text_from_pdf, iter_pdf_documents
from src.ingestion.chunkers import chunk_document_structured, split_into_sentences
from src.ingestion.cache import (
    compute_cache_key,
    compute_chunking_regime_hash,
    load_cached_chunks,
    save_cached_chunks,
    load_dataset_manifest,
    save_dataset_manifest_atomic,
    atomic_write_parquet_table,
)
from src.ingestion.storage import (
    BaseChunkStore,
    InMemoryChunkStore,
    ParquetChunkStore,
    CHUNK_PYARROW_SCHEMA,
    chunks_to_pyarrow_table,
    pyarrow_row_to_chunk,
)
from src.ingestion.pipeline import (
    IngestionPipeline,
    IngestionRegimeMismatchError,
    parse_formatted_chunk_string,
)

__all__ = [
    "extract_text_from_pdf",
    "iter_pdf_documents",
    "chunk_document_structured",
    "split_into_sentences",
    "compute_cache_key",
    "compute_chunking_regime_hash",
    "load_cached_chunks",
    "save_cached_chunks",
    "load_dataset_manifest",
    "save_dataset_manifest_atomic",
    "atomic_write_parquet_table",
    "BaseChunkStore",
    "InMemoryChunkStore",
    "ParquetChunkStore",
    "CHUNK_PYARROW_SCHEMA",
    "chunks_to_pyarrow_table",
    "pyarrow_row_to_chunk",
    "IngestionPipeline",
    "IngestionRegimeMismatchError",
    "parse_formatted_chunk_string",
]
