"""
Generation pipeline package: context building, prompts, generators, and RAG pipeline.
"""

from src.generation.base import BaseGenerator
from src.generation.context import ContextBuilder
from src.generation.factory import get_generator
from src.generation.mock import GroundedSynthesisGenerator
from src.generation.pipeline import GenerationPipeline
from src.generation.prompts import SYSTEM_PROMPT, format_qa_prompt
from src.generation.providers import (
    AnthropicGenerator,
    GeminiGenerator,
    OpenAIGenerator,
)

__all__ = [
    "GenerationPipeline",
    "ContextBuilder",
    "SYSTEM_PROMPT",
    "format_qa_prompt",
    "BaseGenerator",
    "GroundedSynthesisGenerator",
    "OpenAIGenerator",
    "GeminiGenerator",
    "AnthropicGenerator",
    "get_generator",
]
