"""
Generation pipeline package: context building, prompts, generators, and RAG pipeline.
"""

from src.generation.pipeline import GenerationPipeline
from src.generation.context import ContextBuilder
from src.generation.prompts import SYSTEM_PROMPT, format_qa_prompt
from src.generation.base import BaseGenerator
from src.generation.mock import GroundedSynthesisGenerator

__all__ = [
    "GenerationPipeline",
    "ContextBuilder",
    "SYSTEM_PROMPT",
    "format_qa_prompt",
    "BaseGenerator",
    "GroundedSynthesisGenerator",
]
