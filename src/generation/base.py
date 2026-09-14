"""
Abstract Base Class for LLM Generators in the Generation Pipeline.
"""

from abc import ABC, abstractmethod
from typing import List
from src.common.types import DocumentChunk, GenerationResult


class BaseGenerator(ABC):
    """Abstract interface for LLM response generation."""

    @abstractmethod
    def generate(
        self,
        query: str,
        formatted_context: str,
        retrieved_chunks: List[DocumentChunk],
        strategy_used: str = "unknown"
    ) -> GenerationResult:
        """
        Generates a grounded response given query, formatted context, and retrieved chunks.
        """
        pass
