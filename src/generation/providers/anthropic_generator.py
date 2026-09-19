"""
Anthropic LLM Generator Adapter.
Supports direct API (api.anthropic.com) and AWS Bedrock deployment routes.
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
class AnthropicGeneratorConfig:
    """Configuration for Anthropic generator."""

    route: Literal["direct", "bedrock"]
    model: str
    max_tokens: int
    max_retries: int
    retry_base_delay: float
    retry_max_delay: float
    rate_limiter: ProviderRateLimiter


class AnthropicGenerator(BaseGenerator):
    """
    Anthropic Claude generator supporting direct API and AWS Bedrock.

    Direct route uses anthropic SDK with ANTHROPIC_API_KEY.
    Bedrock route uses boto3 with AWS credential chain.
    """

    def __init__(
        self,
        route: Literal["direct", "bedrock"] = "direct",
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
        elif route == "bedrock":
            self._init_bedrock_client()

    def _build_config(
        self,
        route: str,
        model: Optional[str],
        max_tokens: int,
        max_retries: int,
        retry_base_delay: float,
        retry_max_delay: float,
        rate_limiter: Optional[ProviderRateLimiter],
    ) -> AnthropicGeneratorConfig:
        """Build configuration from defaults and environment."""
        provider_config = get_provider_config("anthropic", route)  # type: ignore[arg-type]

        if rate_limiter is None:
            rate_limiter = ProviderRateLimiter(
                rpm=provider_config.rpm,
                input_tpm=provider_config.input_tpm,
                output_tpm=provider_config.output_tpm,
            )

        return AnthropicGeneratorConfig(
            route=route,  # type: ignore[arg-type]
            model=normalize_model_name("anthropic", model or provider_config.model),
            max_tokens=max_tokens,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
            retry_max_delay=retry_max_delay,
            rate_limiter=rate_limiter,
        )

    def _init_direct_client(self):
        """Initialize Anthropic direct API client."""
        try:
            import anthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise AuthError(
                    "ANTHROPIC_API_KEY environment variable not set",
                    provider="anthropic",
                    route="direct",
                )
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
            )

    def _init_bedrock_client(self):
        """Initialize AWS Bedrock client for Anthropic."""
        try:
            import boto3

            self.bedrock_client = boto3.client("bedrock-runtime")
            self.bedrock_region = os.environ.get("AWS_REGION", "us-east-1")
        except ImportError:
            raise ImportError("boto3 package not installed. Install with: pip install boto3")

    def generate(
        self,
        query: str,
        formatted_context: str,
        retrieved_chunks: List[DocumentChunk],
        strategy_used: str = "unknown",
    ) -> GenerationResult:
        """
        Generate a grounded response using Anthropic Claude.

        Implements retry logic with exponential backoff and provider-specific
        error handling for rate limits and capacity issues.
        """
        if not retrieved_chunks:
            return GenerationResult(
                query=query,
                answer="No relevant context passages were retrieved to answer the question.",
                citations=[],
                strategy_used=strategy_used,
                citation_confidence=[],
            )

        # Build prompt
        from src.generation.prompts import SYSTEM_PROMPT, format_qa_prompt

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
                provider="anthropic",
                route=self.route,
            )

        # Retry loop with exponential backoff
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                if self.route == "direct":
                    response = self._call_direct_api(SYSTEM_PROMPT, user_prompt)
                else:
                    response = self._call_bedrock_api(SYSTEM_PROMPT, user_prompt)

                # Calculate per-citation confidence (C3 - retrieval-native signal)
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
                # Use provider-suggested retry-after if available, otherwise exponential backoff
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
            f"Anthropic generation failed after {self.config.max_retries} retries",
            provider="anthropic",
            route=self.route,
            original_error=last_error,
        )

    def _call_direct_api(self, system_prompt: str, user_prompt: str) -> str:
        """Call Anthropic direct API with error handling."""
        try:
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text_blocks = [
                b.text
                for b in (response.content or [])
                if getattr(b, "type", None) == "text" and hasattr(b, "text")
            ]
            return "\n".join(text_blocks) if text_blocks else "[non-text response]"

        except Exception as e:
            self._handle_direct_error(e)

    def _handle_direct_error(self, error: Exception) -> NoReturn:
        """Parse Anthropic direct API errors and raise normalized exceptions."""
        import anthropic

        if isinstance(error, anthropic.RateLimitError):
            # Parse rate limit headers to determine limit type
            limit_type = "unknown"
            retry_after = None

            if hasattr(error, "response") and error.response:
                headers = error.response.headers
                # Check for specific rate limit headers
                if "anthropic-ratelimit-requests-remaining" in headers:
                    limit_type = "rpm"
                elif "anthropic-ratelimit-input-tokens-remaining" in headers:
                    limit_type = "input_tpm"
                elif "anthropic-ratelimit-output-tokens-remaining" in headers:
                    limit_type = "output_tpm"

                # Check for retry-after header
                if "retry-after" in headers:
                    try:
                        retry_after = float(headers["retry-after"])
                    except (ValueError, TypeError):
                        pass

            raise RateLimitError(
                f"Anthropic rate limit exceeded: {limit_type}",
                limit_type=limit_type,
                retry_after=retry_after,
                provider="anthropic",
                route="direct",
                original_error=error,
            )

        elif isinstance(error, anthropic.APIStatusError):
            # Check for 503 capacity errors
            if hasattr(error, "status") and error.status == 503:
                retry_after = None
                if hasattr(error, "response") and error.response:
                    headers = error.response.headers
                    if "retry-after" in headers:
                        try:
                            retry_after = float(headers["retry-after"])
                        except (ValueError, TypeError):
                            pass

                raise CapacityError(
                    "Anthropic API is at capacity (503)",
                    retry_after=retry_after,
                    provider="anthropic",
                    route="direct",
                    original_error=error,
                )

        elif isinstance(error, anthropic.AuthenticationError):
            raise AuthError(
                f"Anthropic authentication failed: {error}",
                provider="anthropic",
                route="direct",
                original_error=error,
            )

        # Unknown error - wrap and re-raise
        raise ProviderError(
            f"Anthropic API error: {error}",
            provider="anthropic",
            route="direct",
            original_error=error,
        )

    def _call_bedrock_api(self, system_prompt: str, user_prompt: str) -> str:
        """Call Anthropic via AWS Bedrock with error handling."""
        import json

        try:
            # Bedrock uses a different request format
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.config.max_tokens,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            }

            response = self.bedrock_client.invoke_model(
                modelId=self.config.model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload),
            )

            response_body = json.loads(response["body"].read())
            content = response_body.get("content", [])
            text_blocks = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text" and b.get("text")
            ]
            return "\n".join(text_blocks) if text_blocks else "[non-text response]"

        except Exception as e:
            self._handle_bedrock_error(e)

    def _handle_bedrock_error(self, error: Exception) -> NoReturn:
        """Parse Bedrock errors and raise normalized exceptions."""
        import botocore.exceptions

        if isinstance(error, botocore.exceptions.ClientError):
            error_code = error.response.get("Error", {}).get("Code", "")
            error_message = error.response.get("Error", {}).get("Message", "")

            if error_code == "ThrottlingException":
                # Bedrock throttling - no retry-after header typically
                raise RateLimitError(
                    f"AWS Bedrock throttling: {error_message}",
                    limit_type="unknown",
                    retry_after=None,
                    provider="anthropic",
                    route="bedrock",
                    original_error=error,
                )

            elif error_code == "ServiceQuotaExceededException":
                raise RateLimitError(
                    f"AWS Bedrock service quota exceeded: {error_message}",
                    limit_type="unknown",
                    retry_after=None,
                    provider="anthropic",
                    route="bedrock",
                    original_error=error,
                )

            elif error_code in [
                "UnrecognizedClientException",
                "InvalidSignatureException",
            ]:
                raise AuthError(
                    f"AWS Bedrock authentication failed: {error_message}",
                    provider="anthropic",
                    route="bedrock",
                    original_error=error,
                )

            elif error_code == "ServiceUnavailableException":
                raise CapacityError(
                    f"AWS Bedrock service unavailable: {error_message}",
                    retry_after=None,
                    provider="anthropic",
                    route="bedrock",
                    original_error=error,
                )

        # Unknown error - wrap and re-raise
        raise ProviderError(
            f"AWS Bedrock error: {error}",
            provider="anthropic",
            route="bedrock",
            original_error=error,
        )

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter."""
        base_delay = self.config.retry_base_delay * (2**attempt)
        jitter = random.uniform(0, base_delay * 0.1)  # 10% jitter
        delay = min(base_delay + jitter, self.config.retry_max_delay)
        return delay
