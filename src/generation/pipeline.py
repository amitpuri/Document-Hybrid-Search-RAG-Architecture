"""
Generation Pipeline connecting retrieved search results to grounded response generation.
"""

from typing import List, Optional, Any
from src.common.types import SearchResult, DocumentChunk, GenerationResult
from src.generation.context import ContextBuilder
from src.generation.prompts import format_qa_prompt
from src.generation.base import BaseGenerator
from src.generation.mock import GroundedSynthesisGenerator


class GenerationPipeline:
    """
    RAG Generation Pipeline:
    1. Formats retrieved chunks into context windows with citations
    2. Builds instruction prompts
    3. Invokes generator model
    4. Returns structured GenerationResult
    """

    def __init__(
        self,
        generator: Optional[BaseGenerator] = None,
        context_builder: Optional[ContextBuilder] = None
    ):
        self.generator = generator or GroundedSynthesisGenerator()
        self.context_builder = context_builder or ContextBuilder()

    def generate_from_results(
        self,
        query: str,
        results: List[SearchResult],
        strategy_used: str = "unknown",
        graph_triplets: Optional[List[Any]] = None,
    ) -> GenerationResult:
        """Constructs context from SearchResult objects and generates a response."""
        chunks = [r.chunk for r in results]
        formatted_context = self.context_builder.build_context(chunks, graph_triplets=graph_triplets)
        return self.generator.generate(
            query=query,
            formatted_context=formatted_context,
            retrieved_chunks=chunks,
            strategy_used=strategy_used
        )

    def generate_from_chunks(
        self,
        query: str,
        chunks: List[DocumentChunk],
        strategy_used: str = "unknown",
        graph_triplets: Optional[List[Any]] = None,
    ) -> GenerationResult:
        """Constructs context directly from DocumentChunk objects and generates a response."""
        formatted_context = self.context_builder.build_context(chunks, graph_triplets=graph_triplets)
        return self.generator.generate(
            query=query,
            formatted_context=formatted_context,
            retrieved_chunks=chunks,
            strategy_used=strategy_used
        )

