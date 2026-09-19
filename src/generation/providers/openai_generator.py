"""
OpenAI LLM Generator Adapter.
Supports direct API (api.openai.com) and Azure OpenAI deployment routes.
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
class OpenAIGeneratorConfig:
    """Configuration for OpenAI generator."""

    route: Literal["direct", "azure"]
    model: str
    max_tokens: int
    max_retries: int
    retry_base_delay: float
    retry_max_delay: float
    rate_limiter: ProviderRateLimiter
    # Azure-specific
    azure_endpoint: Optional[str] = None
    azure_api_version: Optional[str] = None


class OpenAIGenerator(BaseGenerator):
    """
    OpenAI GPT generator supporting direct API and Azure OpenAI.

    Direct route uses openai SDK with OPENAI_API_KEY.
    Azure route uses AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT.
    """

    def __init__(
        self,
        route: Literal["direct", "azure"] = "direct",
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
        elif route == "azure":
            self._init_azure_client()

    def _build_config(
        self,
        route: str,
        model: Optional[str],
        max_tokens: int,
        max_retries: int,
        retry_base_delay: float,
        retry_max_delay: float,
        rate_limiter: Optional[ProviderRateLimiter],
    ) -> OpenAIGeneratorConfig:
        """Build configuration from defaults and environment."""
        provider_config = get_provider_config("openai", route)  # type: ignore[arg-type]

        if rate_limiter is None:
            rate_limiter = ProviderRateLimiter(
                rpm=provider_config.rpm,
                input_tpm=provider_config.input_tpm,
                output_tpm=provider_config.output_tpm,
            )

        azure_endpoint = None
        azure_api_version = None
        if route == "azure":
            azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

            if not azure_endpoint:
                raise AuthError(
                    "AZURE_OPENAI_ENDPOINT environment variable not set",
                    provider="openai",
                    route="azure",
                )

        return OpenAIGeneratorConfig(
            route=route,  # type: ignore[arg-type]
            model=normalize_model_name("openai", model or provider_config.model),
            max_tokens=max_tokens,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
            retry_max_delay=retry_max_delay,
            rate_limiter=rate_limiter,
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_api_version,
        )

    def _init_direct_client(self):
        """Initialize OpenAI direct API client."""
        try:
            import openai

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise AuthError(
                    "OPENAI_API_KEY environment variable not set",
                    provider="openai",
                    route="direct",
                )
            self.client = openai.OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package not installed. " "Install with: pip install openai")

    def _init_azure_client(self):
        """Initialize Azure OpenAI client."""
        try:
            import openai

            api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            if not api_key:
                raise AuthError(
                    "AZURE_OPENAI_API_KEY environment variable not set",
                    provider="openai",
                    route="azure",
                )
            self.client = openai.AzureOpenAI(
                api_key=api_key,
                azure_endpoint=self.config.azure_endpoint,
                api_version=self.config.azure_api_version,
            )
        except ImportError:
            raise ImportError("openai package not installed. " "Install with: pip install openai")

    def generate(
        self,
        query: str,
        formatted_context: str,
        retrieved_chunks: List[DocumentChunk],
        strategy_used: str = "unknown",
    ) -> GenerationResult:
        """
        Generate a grounded response using OpenAI GPT.

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

        # Build prompt
        user_prompt = format_qa_prompt(query, formatted_context)

        # Acquire rate limit quota
        est_input_tokens = len(user_prompt) // 4  # Rough estimate
        est_output_tokens = self.config.max_tokens // 2  # Conservative estimate

        if not self.config.rate_limiter.acquire(
            est_input_tokens=est_input_tokens,
            est_output_tokens=est_output_tokens,
            timeout=10.0,
        ):
            raise RateLimitError(
                "Rate limit quota acquisition timed out",
                provider="openai",
                route=self.route,
            )

        # Retry loop with exponential backoff
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = self._call_api(SYSTEM_PROMPT, user_prompt)

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
            "OpenAI generation failed after max retries",
            provider="openai",
            route=self.route,
            original_error=last_error,
        )

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI API with error handling."""
        try:
            # GPT-5+ and reasoning models require
            # max_completion_tokens instead of max_tokens
            use_completion_tokens = (
                self.config.model.startswith("gpt-5")
                or self.config.model.startswith("gpt-6")
                or self.config.model.startswith("o1")
                or self.config.model.startswith("o3")
            )

            call_kwargs = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            if use_completion_tokens:
                max_tokens = self.config.max_tokens
                call_kwargs["max_completion_tokens"] = max_tokens  # type: ignore[assignment]
            else:
                max_tokens = self.config.max_tokens
                call_kwargs["max_tokens"] = max_tokens  # type: ignore[assignment]

            response = self.client.chat.completions.create(**call_kwargs)

            if not response.choices:
                raise ValueError("Empty response choices from OpenAI API")
            msg = response.choices[0].message
            content = msg.content or ""
            # When an OpenAI model issues a refusal, msg.content is empty and
            # msg.refusal contains the explanation.
            if not content and hasattr(msg, "refusal") and msg.refusal:
                return f"[Model declined to answer: {msg.refusal}]"
            if not content:
                raise ValueError("Empty response from OpenAI API")
            return content

        except Exception as e:
            self._handle_error(e)

    def _handle_error(self, error: Exception) -> NoReturn:
        """Parse OpenAI errors and raise normalized exceptions."""
        import openai

        if isinstance(error, openai.RateLimitError):
            # Distinguish between ramp-rate and capacity errors
            error_message = str(error)
            limit_type = "unknown"
            retry_after = None

            # Check for ramp-rate vs capacity
            if "slow_down" in error_message.lower() or "ramp" in error_message.lower():
                limit_type = "rpm_ramp"
            elif "token" in error_message.lower():
                limit_type = "tpm"

            # Check for retry-after header
            if hasattr(error, "response") and error.response:
                headers = error.response.headers
                if "retry-after" in headers:
                    try:
                        retry_after = float(headers["retry-after"])
                    except (ValueError, TypeError):
                        pass

            raise RateLimitError(
                f"OpenAI rate limit exceeded: {limit_type}",
                limit_type=limit_type,
                retry_after=retry_after,
                provider="openai",
                route=self.route,
                original_error=error,
            )

        elif isinstance(error, openai.APIStatusError):
            status_code = error.status_code if hasattr(error, "status_code") else None

            if status_code == 503:
                retry_after = None
                if hasattr(error, "response") and error.response:
                    headers = error.response.headers
                    if "retry-after" in headers:
                        try:
                            retry_after = float(headers["retry-after"])
                        except (ValueError, TypeError):
                            pass

                raise CapacityError(
                    "OpenAI API is at capacity (503)",
                    retry_after=retry_after,
                    provider="openai",
                    route=self.route,
                    original_error=error,
                )

            elif status_code == 429:
                # 429 that's not caught by RateLimitError above
                retry_after = None
                if hasattr(error, "response") and error.response:
                    headers = error.response.headers
                    if "retry-after" in headers:
                        try:
                            retry_after = float(headers["retry-after"])
                        except (ValueError, TypeError):
                            pass

                raise RateLimitError(
                    f"OpenAI rate limit (429): {error}",
                    limit_type="unknown",
                    retry_after=retry_after,
                    provider="openai",
                    route=self.route,
                    original_error=error,
                )

        elif isinstance(error, openai.AuthenticationError):
            raise AuthError(
                f"OpenAI authentication failed: {error}",
                provider="openai",
                route=self.route,
                original_error=error,
            )

        # Unknown error - wrap and re-raise
        raise ProviderError(
            f"OpenAI API error: {error}",
            provider="openai",
            route=self.route,
            original_error=error,
        )

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter."""
        base_delay = self.config.retry_base_delay * (2**attempt)
        jitter = random.uniform(0, base_delay * 0.1)  # 10% jitter
        delay = min(base_delay + jitter, self.config.retry_max_delay)
        return delay
