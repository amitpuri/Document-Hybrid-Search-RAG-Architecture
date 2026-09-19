"""
Ingestion pipeline package: extractors, chunkers, caching, storage, and pipeline orchestrator.
"""

from src.ingestion.cache import (
    atomic_write_parquet_table,
    compute_cache_key,
    compute_chunking_regime_hash,
    load_cached_chunks,
    load_dataset_manifest,
    save_cached_chunks,
    save_dataset_manifest_atomic,
)
from src.ingestion.chunkers import (
    chunk_document_structured,
    split_into_sentences,
)
from src.ingestion.extractors import extract_text_from_pdf, iter_pdf_documents
from src.ingestion.pipeline import (
    IngestionPipeline,
    IngestionRegimeMismatchError,
    parse_formatted_chunk_string,
)
from src.ingestion.storage import (
    CHUNK_PYARROW_SCHEMA,
    BaseChunkStore,
    InMemoryChunkStore,
    ParquetChunkStore,
    chunks_to_pyarrow_table,
    pyarrow_row_to_chunk,
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
