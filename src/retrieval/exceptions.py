"""
Retrieval Pipeline Exception Classes.

Provides specific exception types for retrieval-related errors to enable
better error handling and debugging across the retrieval pipeline.
"""

from typing import Optional


class RetrievalError(Exception):
    """Base exception for all retrieval-related errors."""

    def __init__(
        self,
        message: str,
        strategy: str = "",
        original_error: Optional[Exception] = None,
    ):
        self.strategy = strategy
        self.original_error = original_error
        super().__init__(message)


class IndexingError(RetrievalError):
    """Raised when indexing operations fail."""

    def __init__(
        self,
        message: str,
        retriever_type: str = "",
        strategy: str = "",
        original_error: Optional[Exception] = None,
    ):
        self.retriever_type = retriever_type
        super().__init__(message, strategy, original_error)


class ScoringError(RetrievalError):
    """Raised when scoring operations fail."""

    def __init__(
        self,
        message: str,
        query: str = "",
        retriever_type: str = "",
        strategy: str = "",
        original_error: Optional[Exception] = None,
    ):
        self.query = query
        self.retriever_type = retriever_type
        super().__init__(message, strategy, original_error)


class ModelLoadError(RetrievalError):
    """Raised when neural model loading fails."""

    def __init__(
        self,
        message: str,
        model_name: str = "",
        retriever_type: str = "",
        strategy: str = "",
        original_error: Optional[Exception] = None,
    ):
        self.model_name = model_name
        self.retriever_type = retriever_type
        super().__init__(message, strategy, original_error)


class FusionError(RetrievalError):
    """Raised when fusion operations fail."""

    def __init__(
        self,
        message: str,
        fusion_method: str = "",
        strategy: str = "",
        original_error: Optional[Exception] = None,
    ):
        self.fusion_method = fusion_method
        super().__init__(message, strategy, original_error)


class GraphRetrievalError(RetrievalError):
    """Raised when graph retrieval operations fail."""

    def __init__(
        self,
        message: str,
        entity: str = "",
        strategy: str = "",
        original_error: Optional[Exception] = None,
    ):
        self.entity = entity
        super().__init__(message, strategy, original_error)


class StrategyNotFoundError(RetrievalError):
    """Raised when a requested strategy is not available."""

    def __init__(
        self,
        message: str,
        requested_strategy: str = "",
        available_strategies: Optional[list] = None,
        original_error: Optional[Exception] = None,
    ):
        self.requested_strategy = requested_strategy
        self.available_strategies = available_strategies or []
        super().__init__(message, "", original_error)
