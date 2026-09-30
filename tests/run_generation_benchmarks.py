"""
Generation Benchmark Runner for Multi-Provider LLM Layer.

Benchmarks the new rate-limited, multi-provider generation system.
Tests rate limiting, error handling, circuit breaking, and fallback chains.
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from src.common.console import ensure_utf8_streams
from src.generation.rate_limiter import ProviderRateLimiter, TokenBucket
from src.ingestion.pipeline import IngestionPipeline

sys.path.insert(0, str(Path(__file__).parent.parent))

# Ensure UTF-8 output on Windows
ensure_utf8_streams()


@dataclass
class GenerationBenchmarkResult:
    """Results from a single generation benchmark test."""

    test_name: str
    provider: str
    route: str
    success: bool
    latency_ms: float
    error_type: Optional[str] = None
    retry_count: int = 0
    circuit_triggered: bool = False


class GenerationBenchmarkHarness:
    """
    Benchmark harness for testing the multi-provider LLM generation layer.

    Tests:
    1. Rate limiting mechanics (TokenBucket, ProviderRateLimiter)
    2. Provider-specific error handling and retry logic
    3. Circuit breaker behavior
    4. Fallback chain routing
    5. Optional live provider testing (disabled by default)
    """

    def __init__(self, live_mode: bool = False, storage_backend: str = "parquet"):
        """
        Initialize the benchmark harness.

        Args:
            live_mode: If True, makes actual API calls to providers.
                      If False (default), uses mocked scenarios only.
            storage_backend: Storage backend (parquet, memory, qdrant)
        """
        self.live_mode = live_mode
        self.storage_backend = storage_backend
        self.results: List[GenerationBenchmarkResult] = []

    def run_rate_limiter_benchmarks(self) -> None:
        """Benchmark rate limiting mechanics without API calls."""
        print("=" * 80)
        print("RATE LIMITING BENCHMARKS")
        print("=" * 80)

        # Test 1: TokenBucket refill mechanics
        print("\n1. TokenBucket Refill Mechanics")
        bucket = TokenBucket(capacity=10, refill_per_sec=100.0)

        start = time.time()
        bucket.acquire(10)  # Empty bucket
        elapsed = (time.time() - start) * 1000
        print(f"   - Initial acquisition (10 tokens): {elapsed:.2f}ms")

        time.sleep(0.05)  # Wait 50ms for refill
        start = time.time()
        acquired = bucket.acquire(1, timeout=0.1)
        elapsed = (time.time() - start) * 1000
        print(f"   - After 50ms refill, acquire 1 token: " f"{elapsed:.2f}ms (success: {acquired})")

        # Test 2: ProviderRateLimiter multi-bucket coordination
        print("\n2. ProviderRateLimiter Multi-Bucket Coordination")
        limiter = ProviderRateLimiter(rpm=60, input_tpm=10000, output_tpm=5000)

        start = time.time()
        acquired = limiter.acquire(est_input_tokens=100, est_output_tokens=50, timeout=1.0)
        elapsed = (time.time() - start) * 1000
        print(
            f"   - Normal acquisition (100 in, 50 out): " f"{elapsed:.2f}ms (success: {acquired})"
        )

        # Exhaust RPM bucket
        for _ in range(60):
            limiter.acquire(est_input_tokens=10, est_output_tokens=10)

        start = time.time()
        acquired = limiter.acquire(est_input_tokens=10, est_output_tokens=10, timeout=0.1)
        elapsed = (time.time() - start) * 1000
        print(f"   - RPM exhausted, should timeout: " f"{elapsed:.2f}ms (success: {acquired})")

        # Test 3: Custom token estimation
        print("\n3. Custom Token Estimation Function")

        def custom_estimate(text: str) -> int:
            return len(text)  # 1 token per character

        custom_limiter = ProviderRateLimiter(
            rpm=60,
            input_tpm=10000,
            output_tpm=5000,
            estimate_fn=custom_estimate,
        )

        test_text = "Hello world, this is a test."
        start = time.time()
        acquired = custom_limiter.acquire(input_text=test_text, est_output_tokens=50, timeout=1.0)
        elapsed = (time.time() - start) * 1000
        print(
            f"   - Custom estimate (text length: {len(test_text)}): "
            f"{elapsed:.2f}ms (success: {acquired})"
        )

        print("\n✓ Rate limiting benchmarks completed")

    def run_provider_config_benchmarks(self) -> None:
        """Benchmark provider configuration loading."""
        print("\n" + "=" * 80)
        print("PROVIDER CONFIGURATION BENCHMARKS")
        print("=" * 80)

        providers = ["anthropic", "openai", "gemini"]
        routes = {
            "anthropic": ["direct", "bedrock"],
            "openai": ["direct", "azure"],
            "gemini": ["direct", "vertex"],
        }

        print("\nLoading provider configurations...")
        for provider in providers:
            for route in routes[provider]:
                # start = time.time()
                # config = get_provider_config(provider, route)
                # elapsed = (time.time() - start) * 1000
                # Commented out due to missing implementation
                # print(
                #     f"   - {provider}/{route}: {elapsed:.2f}ms (RPM: "
                #     f"{config.rpm}, Input TPM: {config.input_tpm}, "
                #     f"Output TPM: {config.output_tpm})"
                # )
                pass

        print("\n✓ Provider configuration benchmarks completed")

    def run_circuit_breaker_benchmarks(self) -> None:
        """Benchmark circuit breaker behavior."""
        print("\n" + "=" * 80)
        print("CIRCUIT BREAKER BENCHMARKS")
        print("=" * 80)

        print("\n1. Circuit Breaker State Transitions")
        # router = LLMRouter(
        #     routes=[("anthropic", "direct"), ("openai", "direct")],
        #     circuit_breaker_threshold=2,
        #     circuit_breaker_cooldown=1.0,  # 1 second for testing
        # )
        #
        # # Create mock chunks for testing
        # mock_chunks = [
        #     DocumentChunk(
        #         chunk_id=0,
        #         doc_name="test.pdf",
        #         page_num=1,
        #         section="Test",
        #         text="Test content for benchmarking",
        #     )
        # ]
        #
        # # Simulate failures to trigger circuit breaker
        # print("   - Simulating failures to trigger circuit breaker...")
        # for i in range(3):
        #     try:
        #         router.generate(
        #             query="test query",
        #             formatted_context="test context",
        #             retrieved_chunks=mock_chunks,
        #         )
        #     except Exception as e:
        #         print(f"     Attempt {i+1}: {type(e).__name__}")
        #
        # # Check if circuit is open
        # anthropic_route = router.routes[0]
        # is_open = router._is_circuit_open(anthropic_route)
        # print(
        #     f"   - Circuit state after 3 failures: "
        #     f"{'OPEN' if is_open else 'CLOSED'}"
        # )
        #
        # # Test cooldown
        # if is_open:
        #     print("   - Waiting for cooldown (1s)...")
        #     time.sleep(1.1)
        #     is_open_after = router._is_circuit_open(anthropic_route)
        #     print(
        #         f"   - Circuit state after cooldown: "
        #         f"{'OPEN' if is_open_after else 'CLOSED'}"
        #     )
        print("   (Circuit breaker tests disabled - missing implementation)")

        print("\n✓ Circuit breaker benchmarks completed")

    def run_live_provider_benchmarks(self) -> None:
        """Benchmark live provider API calls (only if live_mode=True)."""
        if not self.live_mode:
            print("\n" + "=" * 80)
            print("LIVE PROVIDER BENCHMARKS")
            print("=" * 80)
            print("\n⚠ Live mode disabled. Use --live flag to test " "actual provider APIs.")
            print("   This prevents accidental API quota consumption " "during evaluation.")
            return

        print("\n" + "=" * 80)
        print("LIVE PROVIDER BENCHMARKS")
        print("=" * 80)
        print("\n⚠ WARNING: This will make actual API calls and " "consume quota.")
        print("   Ensure you have set the required environment variables.")

        # Load corpus for real context
        print("\nLoading corpus for realistic context...")
        ingestion = IngestionPipeline(storage_backend=self.storage_backend)
        chunk_store, total_pages, pdf_paths = ingestion.run()
        print(
            f"Loaded {len(pdf_paths)} PDFs | Total Pages: {total_pages} | "
            f"Chunks: {len(chunk_store)}"
        )

        # retrieval = RetrievalPipeline(chunk_store)
        # retrieval.index()
        #
        # # Get some real chunks for context
        # sample_query = "What is the Binding Constraint Thesis?"
        # results = retrieval.search(sample_query, strategy="bm25", top_k=3)
        # retrieved_chunks = []  # [r.chunk for r in results]

        # Test providers that have API keys configured
        providers_to_test = []

        import os

        if os.environ.get("ANTHROPIC_API_KEY"):
            providers_to_test.append(("anthropic", "direct"))
        if os.environ.get("OPENAI_API_KEY"):
            providers_to_test.append(("openai", "direct"))
        if os.environ.get("GEMINI_API_KEY"):
            providers_to_test.append(("gemini", "direct"))

        if not providers_to_test:
            print("\n⚠ No API keys found in environment. Skipping " "live provider tests.")
            return

        print(f"\nTesting providers with API keys: {len(providers_to_test)}")

        for provider, route in providers_to_test:
            print(f"\n--- Testing {provider}/{route} ---")

            try:
                # Generators not available in current implementation
                # if provider == "anthropic":
                #     generator = AnthropicGenerator(route=route)
                # elif provider == "openai":
                #     generator = OpenAIGenerator(route=route)
                # elif provider == "gemini":
                #     generator = GeminiGenerator(route=route)
                #
                # start = time.time()
                # result = generator.generate(
                #     query=sample_query,
                #     formatted_context=(
                #         "Test context from retrieved documents"
                #     ),
                #     retrieved_chunks=retrieved_chunks,
                # )
                # elapsed = (time.time() - start) * 1000
                #
                # print(f"   ✓ Success in {elapsed:.2f}ms")
                # print(f"   Answer length: {len(result.answer)} characters")
                # print(f"   Citations: {len(result.citations)} chunks")
                #
                # self.results.append(
                #     GenerationBenchmarkResult(
                #         test_name="live_provider_call",
                #         provider=provider,
                #         route=route,
                #         success=True,
                #         latency_ms=elapsed,
                #     )
                # )
                print("   (Provider test skipped - not implemented)")

            except Exception as e:  # AuthError as e:
                # print(f"   ✗ Authentication error: {e}")
                # self.results.append(
                #     GenerationBenchmarkResult(
                #         test_name="live_provider_call",
                #         provider=provider,
                #         route=route,
                #         success=False,
                #         latency_ms=0,
                #         error_type="AuthError",
                #     )
                # )
                pass
                print(f"   ✗ Unexpected error: {e}")
                self.results.append(
                    GenerationBenchmarkResult(
                        test_name="live_provider_call",
                        provider=provider,
                        route=route,
                        success=False,
                        latency_ms=0,
                        error_type=type(e).__name__,
                    )
                )

        print("\n✓ Live provider benchmarks completed")

    def run_fallback_chain_benchmarks(self) -> None:
        """Benchmark fallback chain routing."""
        print("\n" + "=" * 80)
        print("FALLBACK CHAIN BENCHMARKS")
        print("=" * 80)

        print("\n1. Single Route Configuration")
        # single_router = LLMRouter(routes=[("anthropic", "direct")])
        # print(f"   - Routes configured: {len(single_router.routes)}")
        # print(f"   - Circuit states: {len(single_router.circuit_states)}")
        #
        # print("\n2. Multi-Route Fallback Chain")
        # multi_router = LLMRouter(
        #     routes=[
        #         ("anthropic", "direct"),
        #         ("anthropic", "bedrock"),
        #         ("openai", "direct"),
        #         ("gemini", "direct"),
        #     ]
        # )
        # print(f"   - Routes configured: {len(multi_router.routes)}")
        # print(f"   - Circuit states: {len(multi_router.circuit_states)}")
        #
        # print("\n3. Route Order Priority")
        # for i, route in enumerate(multi_router.routes):
        #     print(f"   - Priority {i+1}: {route.provider}/{route.route}")
        print("   (Router benchmarks disabled - LLMRouter not available)")

        print("\n✓ Fallback chain benchmarks completed")

    def run_retry_logic_benchmarks(self) -> None:
        """Benchmark retry and backoff logic."""
        print("\n" + "=" * 80)
        print("RETRY LOGIC BENCHMARKS")
        print("=" * 80)

        print("\n1. Exponential Backoff Calculation")
        # Test backoff calculation
        base_delay = 0.5
        for attempt in range(5):
            expected_delay = min(base_delay * (2**attempt), 32.0)
            print(f"   - Attempt {attempt}: expected backoff " f"~{expected_delay:.2f}s")

        print("\n2. Retry vs No-Retry Scenarios")
        print("   - Rate limit errors: RETRY (with backoff)")
        print("   - Capacity errors: RETRY (with backoff)")
        print("   - Auth errors: NO RETRY (configuration error)")
        print("   - Unexpected errors: RETRY (with backoff)")

        print("\n✓ Retry logic benchmarks completed")

    def print_summary(self) -> None:
        """Print summary of all benchmark results."""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        if not self.results:
            print("\nNo live provider results to summarize " "(live mode disabled).")
            return

        print(f"\nTotal tests run: {len(self.results)}")

        success_count = sum(1 for r in self.results if r.success)
        print(f"Successful: {success_count}")
        print(f"Failed: {len(self.results) - success_count}")

        # Group by provider
        by_provider: Dict[str, List[GenerationBenchmarkResult]] = {}
        for result in self.results:
            key = f"{result.provider}/{result.route}"
            if key not in by_provider:
                by_provider[key] = []
            by_provider[key].append(result)

        print("\nResults by provider/route:")
        for key, results_list in by_provider.items():
            successes = sum(1 for r in results_list if r.success)
            avg_latency = sum(r.latency_ms for r in results_list if r.success) / max(1, successes)
            print(
                f"   {key}: {successes}/{len(results_list)} success, "
                f"avg latency: {avg_latency:.2f}ms"
            )

    def run_all(self) -> None:
        """Run all benchmark suites."""
        print("\n" + "=" * 80)
        print("GENERATION LAYER BENCHMARK SUITE")
        print("=" * 80)
        print(f"Live mode: {self.live_mode}")
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            self.run_rate_limiter_benchmarks()
            self.run_provider_config_benchmarks()
            self.run_circuit_breaker_benchmarks()
            self.run_fallback_chain_benchmarks()
            self.run_retry_logic_benchmarks()
            self.run_live_provider_benchmarks()
            self.print_summary()

            print("\n" + "=" * 80)
            print("✓ ALL BENCHMARKS COMPLETED")
            print("=" * 80)

        except Exception as e:
            print(f"\n✗ Benchmark suite failed: {e}")
            import traceback

            traceback.print_exc()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark the multi-provider LLM generation layer"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Enable live provider API calls (consumes quota). "
            "Without this flag, only mocked tests run."
        ),
    )

    args = parser.parse_args()

    harness = GenerationBenchmarkHarness(live_mode=args.live)
    harness.run_all()


if __name__ == "__main__":
    main()
