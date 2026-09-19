"""
Google Gemini LLM Generator Adapter.

Supports direct API and GCP Vertex AI deployment routes.
"""

import os
import random
import time
from dataclasses import dataclass
from typing import List, Literal, NoReturn, Optional

from src.common.types import DocumentChunk, GenerationResult
from src.generation.base import BaseGenerator, normalize_model_name
from src.generation.confidence import (
    compute_citation_confidences,
    simulate_fusion_scores,
)
from src.generation.providers.config import get_provider_config
from src.generation.providers.errors import (
    AuthError,
    CapacityError,
    ProviderError,
    RateLimitError,
)
from src.generation.rate_limiter import ProviderRateLimiter


@dataclass
class GeminiGeneratorConfig:
    """Configuration for Gemini generator."""

    route: Literal["direct", "vertex"]
    model: str
    max_tokens: int
    max_retries: int
    retry_base_delay: float
    retry_max_delay: float
    rate_limiter: ProviderRateLimiter
    # Vertex-specific
    project_id: Optional[str] = None
    location: Optional[str] = None


class GeminiGenerator(BaseGenerator):
    """
    Google Gemini generator supporting direct API and GCP Vertex AI.

    Direct route uses google-genai SDK with GEMINI_API_KEY.
    Vertex route uses google-cloud-aiplatform with GCP credential chain.
    """

    def __init__(
        self,
        route: Literal["direct", "vertex"] = "direct",
        model: Optional[str] = None,
        max_tokens: int = 4096,
        max_retries: int = 5,
        retry_base_delay: float = 0.5,
        retry_max_delay: float = 32.0,
        rate_limiter: Optional[ProviderRateLimiter] = None,
    ):
        self.route = route
        self.config = self._build_config(
            route,
            model,
            max_tokens,
            max_retries,
            retry_base_delay,
            retry_max_delay,
            rate_limiter,
        )

        # Initialize clients based on route
        if route == "direct":
            self._init_direct_client()
        elif route == "vertex":
            self._init_vertex_client()

    def _build_config(
        self,
        route: str,
        model: Optional[str],
        max_tokens: int,
        max_retries: int,
        retry_base_delay: float,
        retry_max_delay: float,
        rate_limiter: Optional[ProviderRateLimiter],
    ) -> GeminiGeneratorConfig:
        """Build configuration from defaults and environment."""
        provider_config = get_provider_config("gemini", route)  # type: ignore[arg-type]

        if rate_limiter is None:
            rate_limiter = ProviderRateLimiter(
                rpm=provider_config.rpm,
                input_tpm=provider_config.input_tpm,
                output_tpm=provider_config.output_tpm,
            )

        project_id = None
        location = None
        if route == "vertex":
            project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get(
                "VERTEX_PROJECT_ID"
            )
            location = os.environ.get("GOOGLE_CLOUD_LOCATION") or os.environ.get(
                "VERTEX_LOCATION", "us-central1"
            )

            if not project_id:
                raise AuthError(
                    "GOOGLE_CLOUD_PROJECT or VERTEX_PROJECT_ID not set",
                    provider="gemini",
                    route="vertex",
                )

        return GeminiGeneratorConfig(
            route=route,  # type: ignore[arg-type]
            model=normalize_model_name("gemini", model or provider_config.model),
            max_tokens=max_tokens,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
            retry_max_delay=retry_max_delay,
            rate_limiter=rate_limiter,
            project_id=project_id,
            location=location,
        )

    def _init_direct_client(self):
        """Initialize Gemini direct API client."""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise AuthError(
                "GEMINI_API_KEY environment variable not set",
                provider="gemini",
                route="direct",
            )
        try:
            from google import genai

            self.client = genai.Client(api_key=api_key)
        except ImportError:
            try:
                import google.generativeai as genai

                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel(self.config.model)
            except ImportError:
                raise ImportError(
                    "google-genai package not installed. " "Install with: pip install google-genai"
                )

    def _init_vertex_client(self):
        """Initialize GCP Vertex AI client."""
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel

            vertexai.init(project=self.config.project_id, location=self.config.location)
            self.client = GenerativeModel(self.config.model)
        except ImportError:
            raise ImportError(
                "google-cloud-aiplatform package not installed. "
                "Install with: pip install google-cloud-aiplatform"
            )

    def generate(
        self,
        query: str,
        formatted_context: str,
        retrieved_chunks: List[DocumentChunk],
        strategy_used: str = "unknown",
    ) -> GenerationResult:
        """
        Generate a grounded response using Google Gemini.

        Implements retry logic with exponential backoff and provider-specific
        error handling for rate limits and capacity issues.
        """
        from src.generation.prompts import SYSTEM_PROMPT, format_qa_prompt

        if not retrieved_chunks:
            return GenerationResult(
                query=query,
                answer="No relevant context retrieved.",
                citations=[],
                strategy_used=strategy_used,
                citation_confidence=[],
            )

        # Build prompt (Gemini combines system and user prompts)
        combined_prompt = f"{SYSTEM_PROMPT}\n\n{format_qa_prompt(query, formatted_context)}"

        # Acquire rate limit quota
        est_input_tokens = len(combined_prompt) // 4  # Rough estimate
        est_output_tokens = self.config.max_tokens // 2  # Conservative estimate

        if not self.config.rate_limiter.acquire(
            est_input_tokens=est_input_tokens,
            est_output_tokens=est_output_tokens,
            timeout=10.0,
        ):
            raise RateLimitError(
                "Rate limit quota acquisition timed out",
                provider="gemini",
                route=self.route,
            )

        # Retry loop with exponential backoff
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = self._call_api(combined_prompt)

                # Calculate per-citation confidence
                fusion_scores = simulate_fusion_scores(retrieved_chunks)
                confidences = compute_citation_confidences(
                    query=query,
                    retrieved_chunks=retrieved_chunks,
                    fusion_scores=fusion_scores,
                    top_k=len(retrieved_chunks),
                )

                return GenerationResult(
                    query=query,
                    answer=response,
                    citations=retrieved_chunks,
                    strategy_used=strategy_used,
                    citation_confidence=[c.value for c in confidences],
                )

            except (RateLimitError, CapacityError) as e:
                last_error = e
                # Use provider retry-after or exponential backoff
                delay = e.retry_after if e.retry_after else self._calculate_backoff(attempt)
                time.sleep(delay)
                continue

            except AuthError as e:
                # Auth errors are not retryable
                raise e

            except Exception as e:
                last_error = e  # type: ignore[assignment]
                # Unknown error - retry with backoff
                delay = self._calculate_backoff(attempt)
                time.sleep(delay)
                continue

        # All retries exhausted
        raise ProviderError(
            "Gemini generation failed after max retries",
            provider="gemini",
            route=self.route,
            original_error=last_error,
        )

    def _call_api(self, prompt: str) -> str:
        """Call Gemini API with error handling."""
        try:
            if hasattr(self.client, "models"):
                from google.genai import types as genai_types

                from src.generation.prompts import SYSTEM_PROMPT

                response = self.client.models.generate_content(
                    model=self.config.model,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        max_output_tokens=self.config.max_tokens,
                    ),
                )
            else:
                response = self.client.generate_content(prompt)

            try:
                text = response.text
                if text:
                    return text
            except (ValueError, AttributeError):
                pass

            if hasattr(response, "candidates") and response.candidates:
                finish_reason = getattr(response.candidates[0], "finish_reason", None)
                if finish_reason:
                    return f"[Response blocked: {finish_reason}]"
            return "[non-text or blocked response]"

        except Exception as e:
            self._handle_error(e)

    def _handle_error(self, error: Exception) -> NoReturn:
        """Parse Gemini errors and raise normalized exceptions."""
        error_str = str(error).lower()

        # Check for RESOURCE_EXHAUSTED errors
        if "resource_exhausted" in error_str or "quota" in error_str:
            limit_type = "unknown"
            retry_after = None

            # Try to parse RetryInfo from error message
            if "retrydelay" in error_str or "retry-after" in error_str:
                # Attempt to extract delay from error message
                import re

                delay_match = re.search(r"(\d+(?:\.\d+)?)\s*s[econd]*", error_str)
                if delay_match:
                    try:
                        retry_after = float(delay_match.group(1))
                    except (ValueError, TypeError):
                        pass

            # Determine limit type from error message
            if "rpm" in error_str or "request" in error_str:
                limit_type = "rpm"
            elif "token" in error_str:
                limit_type = "tpm"

            raise RateLimitError(
                f"Gemini quota/resource limit exceeded: {limit_type}",
                limit_type=limit_type,
                retry_after=retry_after,
                provider="gemini",
                route=self.route,
                original_error=error,
            )

        # Check for capacity/server errors
        elif "unavailable" in error_str or "503" in error_str or "overloaded" in error_str:
            retry_after = None
            import re

            delay_match = re.search(r"(\d+(?:\.\d+)?)\s*s[econd]*", error_str)
            if delay_match:
                try:
                    retry_after = float(delay_match.group(1))
                except (ValueError, TypeError):
                    pass

            raise CapacityError(
                "Gemini service unavailable or overloaded",
                retry_after=retry_after,
                provider="gemini",
                route=self.route,
                original_error=error,
            )

        # Check for authentication errors
        elif (
            "unauthenticated" in error_str
            or "permission" in error_str
            or "401" in error_str
            or "403" in error_str
        ):
            raise AuthError(
                "Gemini authentication/authorization failed",
                provider="gemini",
                route=self.route,
                original_error=error,
            )

        # Unknown error - wrap and re-raise
        raise ProviderError(
            f"Gemini API error: {error}",
            provider="gemini",
            route=self.route,
            original_error=error,
        )

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter."""
        base_delay = self.config.retry_base_delay * (2**attempt)
        jitter = random.uniform(0, base_delay * 0.1)  # 10% jitter
        delay = min(base_delay + jitter, self.config.retry_max_delay)
        return delay
