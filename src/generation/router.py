"""
LLM Router for Multi-Provider Fallback and Circuit Breaking.
Manages provider selection, circuit breaking, and graceful degradation.
"""

import logging
import time
from dataclasses import dataclass
from threading import Lock
from typing import List, Literal, Tuple

from src.common.types import DocumentChunk, GenerationResult
from src.generation.base import BaseGenerator
from src.generation.providers import (
    AnthropicGenerator,
    GeminiGenerator,
    OpenAIGenerator,
)
from src.generation.providers.errors import (
    CapacityError,
    ProviderError,
    RateLimitError,
)

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreakerState:
    """Tracks circuit breaker state for a specific route."""

    is_open: bool = False
    failure_count: int = 0
    last_failure_time: float = 0.0
    cooling_until: float = 0.0  # Timestamp when circuit can close


@dataclass
class RouteConfig:
    """Configuration for a single route in the fallback chain."""

    provider: Literal["anthropic", "openai", "gemini"]
    route: Literal["direct", "bedrock", "azure", "vertex"]
    circuit_breaker_threshold: int = 3  # Failures before opening circuit
    circuit_breaker_cooldown: float = 60.0  # Seconds to cool down


class LLMRouter(BaseGenerator):
    """
    Router that manages multiple LLM provider routes with circuit breaking.

    Routes are tried in order; if a route fails due to rate limits or capacity,
    the router advances to the next route. Circuit breakers prevent retrying
    routes that are consistently failing.

    Args:
        routes: Ordered list of (provider, route) tuples to try
        circuit_breaker_threshold: Failures before opening circuit (default: 3)
        circuit_breaker_cooldown: Seconds to cool down before retry (default: 60)
    """

    def __init__(
        self,
        routes: List[
            Tuple[
                Literal["anthropic", "openai", "gemini"],
                Literal["direct", "bedrock", "azure", "vertex"],
            ]
        ],
        circuit_breaker_threshold: int = 3,
        circuit_breaker_cooldown: float = 60.0,
    ):
        self.routes = [
            RouteConfig(p, r, circuit_breaker_threshold, circuit_breaker_cooldown)
            for p, r in routes
        ]
        self.circuit_states: dict = {}  # (provider, route) -> CircuitBreakerState
        self._lock = Lock()

        # Initialize generators for each route
        self.generators: dict = {}
        for route_config in self.routes:
            key = (route_config.provider, route_config.route)
            self.circuit_states[key] = CircuitBreakerState()
            self.generators[key] = self._create_generator(route_config.provider, route_config.route)

    def _create_generator(
        self,
        provider: Literal["anthropic", "openai", "gemini"],
        route: Literal["direct", "bedrock", "azure", "vertex"],
    ) -> BaseGenerator:
        """Create a generator instance for the given provider and route."""
        if provider == "anthropic":
            return AnthropicGenerator(route=route)  # type: ignore[arg-type]
        elif provider == "openai":
            return OpenAIGenerator(route=route)  # type: ignore[arg-type]
        elif provider == "gemini":
            return GeminiGenerator(route=route)  # type: ignore[arg-type]
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _is_circuit_open(self, route_config: RouteConfig) -> bool:
        """Check if circuit breaker is open for a route."""
        key = (route_config.provider, route_config.route)
        state = self.circuit_states[key]

        with self._lock:
            # Check if cooling period has elapsed
            if state.is_open and time.time() >= state.cooling_until:
                # Close circuit and reset
                state.is_open = False
                state.failure_count = 0
                logger.info(f"Circuit closed for {route_config.provider}/{route_config.route}")
                return False

            return state.is_open

    def _record_failure(self, route_config: RouteConfig, error: Exception):
        """Record a failure and potentially open the circuit breaker."""
        key = (route_config.provider, route_config.route)
        state = self.circuit_states[key]

        with self._lock:
            state.failure_count += 1
            state.last_failure_time = time.time()

            # Open circuit if threshold exceeded
            if state.failure_count >= route_config.circuit_breaker_threshold:
                state.is_open = True
                state.cooling_until = time.time() + route_config.circuit_breaker_cooldown
                logger.warning(
                    f"Circuit opened for {route_config.provider}/{route_config.route} "
                    f"after {state.failure_count} failures. Cooling until "
                    f"{state.cooling_until}"
                )

    def _record_success(self, route_config: RouteConfig):
        """Record a success and reset failure count."""
        key = (route_config.provider, route_config.route)
        state = self.circuit_states[key]

        with self._lock:
            state.failure_count = 0
            if state.is_open:
                state.is_open = False
                logger.info(
                    f"Circuit closed for {route_config.provider}/{route_config.route} "
                    f"after success"
                )

    def generate(
        self,
        query: str,
        formatted_context: str,
        retrieved_chunks: List[DocumentChunk],
        strategy_used: str = "unknown",
    ) -> GenerationResult:
        """
        Generate a response using the first available route in the fallback chain.

        Tries routes in order, skipping those with open circuit breakers.
        Returns immediately on success; raises if all routes are exhausted.
        """
        if not retrieved_chunks:
            return GenerationResult(
                query=query,
                answer="No relevant context passages were retrieved to answer the question.",
                citations=[],
                strategy_used=strategy_used,
            )

        last_error = None
        attempted_routes = []

        for route_config in self.routes:
            key = (route_config.provider, route_config.route)

            # Skip if circuit is open
            if self._is_circuit_open(route_config):
                logger.info(
                    f"Skipping {route_config.provider}/{route_config.route} - " f"circuit open"
                )
                continue

            attempted_routes.append(key)
            generator = self.generators[key]

            try:
                logger.info(f"Attempting {route_config.provider}/{route_config.route}")
                result = generator.generate(
                    query=query,
                    formatted_context=formatted_context,
                    retrieved_chunks=retrieved_chunks,
                    strategy_used=strategy_used,
                )

                # Record success
                self._record_success(route_config)
                logger.info(f"Success on {route_config.provider}/{route_config.route}")
                return result

            except (RateLimitError, CapacityError) as e:
                # These are expected transient errors - record failure and try next
                last_error = e
                self._record_failure(route_config, e)
                limit_type = getattr(e, "limit_type", "unknown")
                retry_after = getattr(e, "retry_after", None)
                logger.warning(
                    f"Transient error on {route_config.provider}/{route_config.route}: "
                    f"{e.__class__.__name__} - limit_type={limit_type}, retry_after="
                    f"{retry_after}"
                )
                continue

            except ProviderError as e:
                # Other provider errors - record failure and try next route
                last_error = e  # type: ignore[assignment]
                self._record_failure(route_config, e)
                logger.error(f"Provider error on {route_config.provider}/{route_config.route}: {e}")
                continue

            except Exception as e:
                # Unexpected errors - record failure and try next route
                last_error = e  # type: ignore[assignment]
                self._record_failure(route_config, e)
                logger.error(
                    f"Unexpected error on {route_config.provider}/{route_config.route}: " f"{e}"
                )
                continue

        # All routes exhausted
        error_msg = (
            f"All routes exhausted. Attempted: {attempted_routes}. Last error: "
            f"{last_error.__class__.__name__ if last_error else 'None'}"
        )
        logger.error(error_msg)
        raise ProviderError(error_msg, original_error=last_error)
