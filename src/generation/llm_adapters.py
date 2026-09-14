"""
LLM Generator Adapters for Grounded RAG Answering.

Provides three concrete BaseGenerator subclasses that call real LLM APIs:
  - OpenAIGenerator    (requires: openai>=1.0.0,      OPENAI_API_KEY env var)
                         Default model: gpt-5
  - GeminiGenerator    (requires: google-genai>=1.0.0, GEMINI_API_KEY env var)
                         Default model: gemini-3.8-flash
  - AnthropicGenerator (requires: anthropic>=0.34.0,  ANTHROPIC_API_KEY env var)
                         Default model: claude-sonnet-5

All adapters use the shared SYSTEM_PROMPT and format_qa_prompt from prompts.py,
and return structured GenerationResult objects with exact DocumentChunk citations.

Install the SDK for whichever provider you use:
    pip install openai
    pip install google-genai          # replaces deprecated google-generativeai
    pip install anthropic
"""

import os
from typing import List

from src.generation.base import BaseGenerator
from src.generation.prompts import SYSTEM_PROMPT, format_qa_prompt
from src.common.types import DocumentChunk, GenerationResult


# ── Optional SDK availability flags ─────────────────────────────────────────

try:
    import openai as _openai_sdk
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from google import genai as _genai_sdk
    from google.genai import types as _genai_types
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    import anthropic as _anthropic_sdk
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# ── OpenAI ───────────────────────────────────────────────────────────────────

class OpenAIGenerator(BaseGenerator):
    """
    Grounded RAG generator backed by the OpenAI Chat Completions API.
    Requires: pip install openai  +  OPENAI_API_KEY environment variable.
    """

    def __init__(
        self,
        model: str = "gpt-5",
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ):
        if not HAS_OPENAI:
            raise ImportError(
                "openai package is not installed. Run: pip install openai"
            )
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set."
            )
        self.client = _openai_sdk.OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(
        self,
        query: str,
        formatted_context: str,
        retrieved_chunks: List[DocumentChunk],
        strategy_used: str = "unknown",
    ) -> GenerationResult:
        if not retrieved_chunks:
            return GenerationResult(
                query=query,
                answer="No relevant context passages were retrieved to answer the question.",
                citations=[],
                strategy_used=strategy_used,
            )

        user_prompt = format_qa_prompt(query, formatted_context)

        create_kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.model.startswith(("gpt-5", "o1", "o3")):
            # Reasoning models require max_completion_tokens (not max_tokens)
            # and need a generous budget to produce visible output after internal CoT
            create_kwargs["max_completion_tokens"] = max(self.max_tokens, 8192)
        else:
            create_kwargs["max_tokens"] = self.max_tokens
            create_kwargs["temperature"] = self.temperature

        response = self.client.chat.completions.create(**create_kwargs)
        msg = response.choices[0].message
        answer = msg.content or ""
        if not answer and hasattr(msg, "refusal") and msg.refusal:
            answer = f"[Model declined to answer: {msg.refusal}]"

        return GenerationResult(
            query=query,
            answer=answer,
            citations=retrieved_chunks,
            strategy_used=strategy_used,
        )


# ── Gemini ───────────────────────────────────────────────────────────────────

class GeminiGenerator(BaseGenerator):
    """
    Grounded RAG generator backed by the Google Gemini API.
    Requires: pip install google-genai  +  GEMINI_API_KEY environment variable.
    Uses the modern google-genai SDK (replaces deprecated google-generativeai).
    Default model: gemini-3.8-flash
    """

    _MODEL_ALIASES = {
        # Pro variants normalize to canonical Pro model
        "gemini-3.8-pro": "gemini-3.8-pro",
        "gemini-pro-3.8": "gemini-3.8-pro",
        "gemini-pro": "gemini-3.8-pro",
        "pro": "gemini-3.8-pro",
        # Flash variants normalize to canonical Flash model
        "gemini-3.8-flash": "gemini-3.8-flash",
        "gemini-flash-3.8": "gemini-3.8-flash",
        "gemini-flash": "gemini-3.8-flash",
        "gemini-3.8": "gemini-3.8-flash",
        "flash": "gemini-3.8-flash",
    }

    def __init__(
        self,
        model: str = "gemini-3.8-flash",
        max_output_tokens: int = 1024,
        temperature: float = 0.2,
    ):
        if not HAS_GEMINI:
            raise ImportError(
                "google-genai package is not installed. "
                "Run: pip install google-genai"
            )
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY environment variable is not set."
            )

        model_clean = model.lower().replace("_", "-")
        model = self._MODEL_ALIASES.get(model_clean, model)

        self.model_name = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        # New SDK: instantiate a client (not a module-level configure call)
        self.client = _genai_sdk.Client(api_key=api_key)

    def generate(
        self,
        query: str,
        formatted_context: str,
        retrieved_chunks: List[DocumentChunk],
        strategy_used: str = "unknown",
    ) -> GenerationResult:
        if not retrieved_chunks:
            return GenerationResult(
                query=query,
                answer="No relevant context passages were retrieved to answer the question.",
                citations=[],
                strategy_used=strategy_used,
            )

        user_prompt = format_qa_prompt(query, formatted_context)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=_genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=self.max_output_tokens,
                temperature=self.temperature,
            ),
        )
        answer = response.text or ""

        return GenerationResult(
            query=query,
            answer=answer,
            citations=retrieved_chunks,
            strategy_used=strategy_used,
        )


# ── Anthropic ────────────────────────────────────────────────────────────────

class AnthropicGenerator(BaseGenerator):
    """
    Grounded RAG generator backed by the Anthropic Messages API.
    Requires: pip install anthropic  +  ANTHROPIC_API_KEY environment variable.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ):
        if not HAS_ANTHROPIC:
            raise ImportError(
                "anthropic package is not installed. Run: pip install anthropic"
            )
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable is not set."
            )
        self.client = _anthropic_sdk.Anthropic(api_key=api_key)

        model_clean = model.lower().replace("_", "-")
        if model_clean in ("sonnet-5", "claude-5", "sonnet 5"):
            model = "claude-sonnet-5"

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(
        self,
        query: str,
        formatted_context: str,
        retrieved_chunks: List[DocumentChunk],
        strategy_used: str = "unknown",
    ) -> GenerationResult:
        if not retrieved_chunks:
            return GenerationResult(
                query=query,
                answer="No relevant context passages were retrieved to answer the question.",
                citations=[],
                strategy_used=strategy_used,
            )

        user_prompt = format_qa_prompt(query, formatted_context)

        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        answer = message.content[0].text if message.content else ""

        return GenerationResult(
            query=query,
            answer=answer,
            citations=retrieved_chunks,
            strategy_used=strategy_used,
        )
