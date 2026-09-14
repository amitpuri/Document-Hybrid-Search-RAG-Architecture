"""
Generator Factory — Auto-Detects Available LLM from Environment Variables.

Priority order (first available wins):
  1. Explicit `prefer` argument  →  the named provider
  2. ANTHROPIC_API_KEY set       →  AnthropicGenerator
  3. OPENAI_API_KEY set          →  OpenAIGenerator
  4. GEMINI_API_KEY set          →  GeminiGenerator
  5. No key found                →  GroundedSynthesisGenerator (offline mock)

Usage:
    from src.generation.factory import get_generator

    # Auto-detect from env:
    generator = get_generator()

    # Force a specific provider:
    generator = get_generator(prefer="gemini")

    # Force the offline mock:
    generator = get_generator(prefer="mock")
"""

import os
from src.generation.base import BaseGenerator


def get_generator(prefer: str | None = None) -> BaseGenerator:
    """
    Returns the best available BaseGenerator given installed SDKs and env vars.

    Args:
        prefer: Optional provider hint — one of "openai", "gemini", "anthropic",
                "mock", or None (auto-detect).

    Returns:
        A configured BaseGenerator ready to call generate().
    """
    # Lazy imports to avoid hard import-time failures if SDKs aren't installed
    from src.generation.llm_adapters import (
        HAS_OPENAI, HAS_GEMINI, HAS_ANTHROPIC,
        OpenAIGenerator, GeminiGenerator, AnthropicGenerator,
    )
    from src.generation.mock import GroundedSynthesisGenerator

    prefer_lower = prefer.lower().strip() if prefer else None

    # ── Explicit mock override ────────────────────────────────────────────────
    if prefer_lower == "mock":
        return GroundedSynthesisGenerator()

    # ── Explicit provider request ─────────────────────────────────────────────
    if prefer_lower == "anthropic":
        return AnthropicGenerator()
    if prefer_lower == "openai":
        return OpenAIGenerator()
    if prefer_lower == "gemini":
        return GeminiGenerator()

    # ── Auto-detect: check env vars in priority order ────────────────────────
    if HAS_ANTHROPIC and os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicGenerator()

    if HAS_OPENAI and os.environ.get("OPENAI_API_KEY"):
        return OpenAIGenerator()

    if HAS_GEMINI and os.environ.get("GEMINI_API_KEY"):
        return GeminiGenerator()

    # ── Fallback: offline mock ────────────────────────────────────────────────
    return GroundedSynthesisGenerator()
