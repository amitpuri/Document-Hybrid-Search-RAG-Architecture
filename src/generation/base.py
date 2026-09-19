"""
Abstract Base Class for LLM Generators in the Generation Pipeline.
"""

from abc import ABC, abstractmethod
from typing import List

from src.common.types import DocumentChunk, GenerationResult

PROVIDER_MODEL_ALIASES = {
    "gemini": {
        "gemini-3.8-pro": "gemini-3.8-pro",
        "gemini-pro-3.8": "gemini-3.8-pro",
        "gemini-pro": "gemini-3.8-pro",
        "pro": "gemini-3.8-pro",
        "gemini-3.8-flash": "gemini-3.8-flash",
        "gemini-flash-3.8": "gemini-3.8-flash",
        "gemini-flash": "gemini-3.8-flash",
        "gemini-3.8": "gemini-3.8-flash",
        "flash": "gemini-3.8-flash",
    },
    "anthropic": {
        "claude-sonnet-5": "claude-sonnet-5",
        "sonnet-5": "claude-sonnet-5",
        "claude-5": "claude-sonnet-5",
        "sonnet 5": "claude-sonnet-5",
        "claude-sonnet-4-6": "claude-sonnet-4-6",
        "sonnet-4.6": "claude-sonnet-4-6",
        "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
        "sonnet-3.5": "claude-3-5-sonnet-20241022",
    },
    "openai": {
        "gpt-5.5": "gpt-5.5",
        "gpt5.5": "gpt-5.5",
        "gpt-5": "gpt-5",
        "gpt5": "gpt-5",
        "gpt-4o": "gpt-4o",
        "gpt4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt4o-mini": "gpt-4o-mini",
        "o1": "o1",
        "o3": "o3",
    },
}


def normalize_model_name(provider: str, model: str) -> str:
    """
    Normalizes user-supplied model names and aliases to canonical provider model identifiers.
    Falls back to the input string if no alias matches.
    """
    if not model:
        return model
    provider_key = provider.lower().strip()
    model_clean = model.lower().strip().replace("_", "-")
    aliases = PROVIDER_MODEL_ALIASES.get(provider_key, {})
    return aliases.get(model_clean, aliases.get(model.lower().strip(), model))


class BaseGenerator(ABC):
    """Abstract interface for LLM response generation."""

    @abstractmethod
    def generate(
        self,
        query: str,
        formatted_context: str,
        retrieved_chunks: List[DocumentChunk],
        strategy_used: str = "unknown",
    ) -> GenerationResult:
        """
        Generates a grounded response given query, formatted context, and retrieved chunks.

        All provider implementations must:
        1. Generate grounded answer with bracketed citations
        2. Calculate per-citation confidence using calculate_confidence() from confidence.py
        3. Return GenerationResult with citation_confidence populated (parallel to citations list)

        Confidence calculation uses retrieval-native signals (rank, fusion score, lexical overlap)
        without requiring a second LLM call.
        """
        pass
