"""
Ingestion Pipeline Exception Classes.

Provides specific exception types for ingestion-related errors to enable
better error handling and debugging across the ingestion pipeline.
"""

from typing import Optional


class IngestionError(Exception):
    """Base exception for all ingestion-related errors."""

    def __init__(
        self,
        message: str,
        doc_name: str = "",
        original_error: Optional[Exception] = None,
    ):
        self.doc_name = doc_name
        self.original_error = original_error
        super().__init__(message)


class CacheError(IngestionError):
    """Raised when cache operations fail (load, save, validation)."""

    def __init__(
        self,
        message: str,
        cache_path: str = "",
        doc_name: str = "",
        original_error: Optional[Exception] = None,
    ):
        self.cache_path = cache_path
        super().__init__(message, doc_name, original_error)


class ExtractionError(IngestionError):
    """Raised when PDF text extraction fails."""

    def __init__(
        self,
        message: str,
        pdf_path: str = "",
        page_num: int = 0,
        doc_name: str = "",
        original_error: Optional[Exception] = None,
    ):
        self.pdf_path = pdf_path
        self.page_num = page_num
        super().__init__(message, doc_name, original_error)


class ChunkingError(IngestionError):
    """Raised when text chunking fails."""

    def __init__(
        self,
        message: str,
        doc_name: str = "",
        chunk_id: int = 0,
        original_error: Optional[Exception] = None,
    ):
        self.chunk_id = chunk_id
        super().__init__(message, doc_name, original_error)


class StorageError(IngestionError):
    """Raised when storage operations fail (Parquet, Qdrant, memory)."""

    def __init__(
        self,
        message: str,
        storage_backend: str = "",
        doc_name: str = "",
        original_error: Optional[Exception] = None,
    ):
        self.storage_backend = storage_backend
        super().__init__(message, doc_name, original_error)


class GraphExtractionError(IngestionError):
    """Raised when knowledge graph extraction fails."""

    def __init__(
        self,
        message: str,
        doc_name: str = "",
        chunk_id: int = 0,
        original_error: Optional[Exception] = None,
    ):
        self.chunk_id = chunk_id
        super().__init__(message, doc_name, original_error)


class RegimeMismatchError(IngestionError):
    """Raised when chunking parameters don't match existing dataset."""

    def __init__(
        self,
        message: str,
        saved_params: Optional[dict] = None,
        requested_params: Optional[dict] = None,
        original_error: Optional[Exception] = None,
    ):
        self.saved_params = saved_params or {}
        self.requested_params = requested_params or {}
        super().__init__(message, "", original_error)
