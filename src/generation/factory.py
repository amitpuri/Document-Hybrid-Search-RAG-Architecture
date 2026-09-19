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
from typing import Union

from src.generation.base import BaseGenerator


def get_generator(prefer: Union[str, None] = None) -> BaseGenerator:
    """
    Returns the best available BaseGenerator given installed SDKs and env vars.

    Args:
        prefer: Optional provider hint — one of "openai", "gemini", "anthropic",
                "mock", or None (auto-detect).

    Returns:
        A configured BaseGenerator ready to call generate().
    """
    from src.generation.mock import GroundedSynthesisGenerator

    prefer_lower = prefer.lower().strip() if prefer else None

    # ── Explicit mock override ────────────────────────────────────────────────
    if prefer_lower == "mock":
        return GroundedSynthesisGenerator()

    # ── Explicit provider request ─────────────────────────────────────────────
    if prefer_lower == "anthropic":
        from src.generation.providers import AnthropicGenerator

        return AnthropicGenerator()
    if prefer_lower == "openai":
        from src.generation.providers import OpenAIGenerator

        return OpenAIGenerator()
    if prefer_lower == "gemini":
        from src.generation.providers import GeminiGenerator

        return GeminiGenerator()

    # ── Auto-detect: check env vars in priority order with graceful fallback ──
    # Priority: Anthropic -> OpenAI -> Gemini -> Mock
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from src.generation.providers import AnthropicGenerator

            return AnthropicGenerator()
        except (ImportError, EnvironmentError, Exception):
            pass

    if os.environ.get("OPENAI_API_KEY"):
        try:
            from src.generation.providers import OpenAIGenerator

            return OpenAIGenerator()
        except (ImportError, EnvironmentError, Exception):
            pass

    if os.environ.get("GEMINI_API_KEY"):
        try:
            from src.generation.providers import GeminiGenerator

            return GeminiGenerator()
        except (ImportError, EnvironmentError, Exception):
            pass

    # ── Fallback: offline mock ────────────────────────────────────────────────
    return GroundedSynthesisGenerator()
