"""
Normalized Error Classes for LLM Provider Adapters.
Provides consistent error handling across different providers and routes.
"""

from typing import Optional


class ProviderError(Exception):
    """Base exception for all provider-related errors."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        route: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        self.provider = provider
        self.route = route
        self.original_error = original_error
        super().__init__(message)


class RateLimitError(ProviderError):
    """
    Raised when a rate limit is exceeded.

    Attributes:
        limit_type: Type of limit that was exceeded ('rpm', 'input_tpm', 'output_tpm', 'unknown')
        retry_after: Suggested retry delay in seconds (if provided by API)
    """

    def __init__(
        self,
        message: str,
        limit_type: str = "unknown",
        retry_after: Optional[float] = None,
        provider: Optional[str] = None,
        route: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        self.limit_type = limit_type
        self.retry_after = retry_after
        super().__init__(message, provider, route, original_error)


class CapacityError(ProviderError):
    """
    Raised when the provider is at capacity (503, server overloaded).
    Distinct from rate limits - this is about server capacity, not quota.

    Attributes:
        retry_after: Suggested retry delay in seconds (if provided by API)
    """

    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        provider: Optional[str] = None,
        route: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        self.retry_after = retry_after
        super().__init__(message, provider, route, original_error)


class AuthError(ProviderError):
    """
    Raised when authentication fails (invalid API key, credentials, etc.).
    This is typically a configuration error, not a transient one.
    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        route: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message, provider, route, original_error)
