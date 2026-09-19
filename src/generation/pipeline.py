"""
Generation Pipeline connecting search results to response generation.
"""

from typing import Any, List, Literal, Optional

from src.common.types import DocumentChunk, GenerationResult, SearchResult
from src.generation.base import BaseGenerator
from src.generation.context import ContextBuilder
from src.generation.mock import GroundedSynthesisGenerator
from src.generation.providers import (
    AnthropicGenerator,
    GeminiGenerator,
    OpenAIGenerator,
)
from src.generation.router import LLMRouter


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
        context_builder: Optional[ContextBuilder] = None,
        provider: Optional[Literal["anthropic", "openai", "gemini", "mock"]] = None,
        route: Optional[Literal["direct", "bedrock", "azure", "vertex"]] = None,
        router_config: Optional[List[tuple]] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the generation pipeline.

        Args:
            generator: Custom generator instance (takes precedence over
                provider/route)
            context_builder: Custom context builder
            provider: Provider name if using single provider
                ('anthropic', 'openai', 'gemini', 'mock')
            route: Deployment route if using single provider
                ('direct', 'bedrock', 'azure', 'vertex')
            router_config: List of (provider, route) tuples for fallback chain
            model: Optional model identifier override
        """
        self.context_builder = context_builder or ContextBuilder()

        # Priority: custom generator > router config > single
        # provider/route > mock
        if generator is not None:
            self.generator = generator
        elif router_config is not None:
            self.generator = LLMRouter(routes=router_config)
        elif provider is not None:
            if provider == "mock":
                self.generator = GroundedSynthesisGenerator()
            else:
                # Create single provider generator
                if provider == "anthropic":
                    self.generator = AnthropicGenerator(
                        route=route or "direct", model=model  # type: ignore[arg-type]
                    )
                elif provider == "openai":
                    self.generator = OpenAIGenerator(
                        route=route or "direct", model=model  # type: ignore[arg-type]
                    )
                elif provider == "gemini":
                    self.generator = GeminiGenerator(
                        route=route or "direct", model=model  # type: ignore[arg-type]
                    )
                else:
                    raise ValueError(f"Unknown provider: {provider}")
        else:
            # Default to mock
            self.generator = GroundedSynthesisGenerator()

    def generate_from_results(
        self,
        query: str,
        results: List[SearchResult],
        strategy_used: str = "unknown",
        graph_triplets: Optional[List[Any]] = None,
    ) -> GenerationResult:
        """Construct context from SearchResult and generate response."""
        chunks = [r.chunk for r in results]
        formatted_context = self.context_builder.build_context(
            chunks, graph_triplets=graph_triplets
        )
        return self.generator.generate(
            query=query,
            formatted_context=formatted_context,
            retrieved_chunks=chunks,
            strategy_used=strategy_used,
        )

    def generate_from_chunks(
        self,
        query: str,
        chunks: List[DocumentChunk],
        strategy_used: str = "unknown",
        graph_triplets: Optional[List[Any]] = None,
    ) -> GenerationResult:
        """Construct context from DocumentChunk and generate response."""
        formatted_context = self.context_builder.build_context(
            chunks, graph_triplets=graph_triplets
        )
        return self.generator.generate(
            query=query,
            formatted_context=formatted_context,
            retrieved_chunks=chunks,
            strategy_used=strategy_used,
        )
