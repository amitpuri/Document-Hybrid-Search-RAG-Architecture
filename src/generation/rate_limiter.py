"""
Rate Limiting for LLM Provider API Calls.
Implements token bucket rate limiting with configurable RPM/TPM limits.
"""

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class TokenBucket:
    """
    Thread-safe token bucket for rate limiting.

    Args:
        capacity: Maximum number of tokens the bucket can hold
        refill_per_sec: Rate at which tokens are refilled (tokens per second)
    """

    capacity: float
    refill_per_sec: float

    def __post_init__(self):
        self._tokens = float(self.capacity)
        self._last_refill = time.time()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time since last refill."""
        now = time.time()
        elapsed = now - self._last_refill
        if elapsed > 0:
            refill_amount = elapsed * self.refill_per_sec
            self._tokens = min(self.capacity, self._tokens + refill_amount)
            self._last_refill = now

    def acquire(self, amount: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Attempt to acquire tokens from the bucket.

        Args:
            amount: Number of tokens to acquire
            timeout: Maximum time to wait for tokens (None = wait indefinitely)

        Returns:
            True if tokens were acquired, False if timeout was reached
        """
        deadline = time.time() + timeout if timeout is not None else None

        while True:
            with self._lock:
                self._refill()
                if self._tokens >= amount:
                    self._tokens -= amount
                    return True

            # Check timeout
            if deadline is not None and time.time() >= deadline:
                return False

            # Wait a bit before retrying
            time.sleep(0.01)


class ProviderRateLimiter:
    """
    Rate limiter for LLM providers tracking RPM, input TPM, and output TPM.

    Args:
        rpm: Requests per minute limit
        input_tpm: Input tokens per minute limit
        output_tpm: Output tokens per minute limit
        estimate_fn: Function to estimate token count from text (default: chars/4)
    """

    def __init__(
        self,
        rpm: int,
        input_tpm: int,
        output_tpm: int,
        estimate_fn: Optional[Callable[[str], int]] = None,
    ):
        self.rpm_bucket = TokenBucket(capacity=rpm, refill_per_sec=rpm / 60.0)
        self.input_tpm_bucket = TokenBucket(capacity=input_tpm, refill_per_sec=input_tpm / 60.0)
        self.output_tpm_bucket = TokenBucket(capacity=output_tpm, refill_per_sec=output_tpm / 60.0)

        self.estimate_fn = estimate_fn or self._default_estimate

    @staticmethod
    def _default_estimate(text: str) -> int:
        """Default token estimation: roughly 4 characters per token."""
        return max(1, len(text) // 4)

    def acquire(
        self,
        est_input_tokens: Optional[int] = None,
        est_output_tokens: Optional[int] = None,
        input_text: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Attempt to acquire quota for a request.

        Args:
            est_input_tokens: Pre-calculated input token count (takes precedence over input_text)
            est_output_tokens: Estimated output token count
            input_text: Input text to estimate tokens from (if est_input_tokens not provided)
            timeout: Maximum time to wait for quota

        Returns:
            True if quota was acquired, False if timeout was reached
        """
        # Estimate input tokens if not provided
        if est_input_tokens is None and input_text is not None:
            est_input_tokens = self.estimate_fn(input_text)
        elif est_input_tokens is None:
            est_input_tokens = 1

        if est_output_tokens is None:
            est_output_tokens = 1

        # Try to acquire from all buckets
        acquired = (
            self.rpm_bucket.acquire(1, timeout)
            and self.input_tpm_bucket.acquire(est_input_tokens, timeout)
            and self.output_tpm_bucket.acquire(est_output_tokens, timeout)
        )

        return acquired
