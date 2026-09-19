"""
Enhanced Benchmark Harness for Document Hybrid Search System.
Provides comprehensive testing infrastructure for retrieval strategies,
generation pipelines, and end-to-end system performance.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))  # noqa: E402

from src.common.console import ensure_utf8_streams  # noqa: E402

# Ensure UTF-8 output on Windows
ensure_utf8_streams()


@dataclass
class BenchmarkResult:
    """Result from a single benchmark execution."""

    query_label: str
    query: str
    strategy: str
    success: bool
    duration_ms: float
    results_count: int
    error_message: Optional[str] = None
    output: Optional[str] = None


class BenchmarkHarness:
    """
    Enhanced benchmark harness for testing retrieval strategies and generation pipelines.

    Features:
    - Configurable strategy and query selection
    - Performance timing and metrics collection
    - Error handling and reporting
    - Output filtering and formatting
    - JSON/Text output options
    """

    def __init__(
        self,
        strategies: Optional[List[str]] = None,
        queries: Optional[List[Tuple[str, str]]] = None,
        top_k: int = 3,
        corpus: str = "corpus",
        storage: str = "parquet",
        verbose: bool = True,
    ):
        self.strategies = strategies or self._get_default_strategies()
        self.queries = queries or self._get_default_queries()
        self.top_k = top_k
        self.corpus = corpus
        self.storage = storage
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []

    def _get_default_strategies(self) -> List[str]:
        """Get default list of retrieval strategies."""
        return [
            "bm25",
            "tfidf",
            "linear_0.3",
            "linear_0.5",
            "linear_0.7",
            "rrf",
            "rrf_dedup",
            "rrf_dedup_mmr",
            "ppmi",
            "cross_encoder",
            "sentence_transformer",
            "adaptive",
            "specter2",
        ]

    def _get_default_queries(self) -> List[Tuple[str, str]]:
        """Get default benchmark queries from README (real corpus queries)."""
        return [
            (
                "a",
                "How does the Binding Constraint Thesis affect harness comparisons?",
            ),
            (
                "b",
                "Dynamic Tiered AgentRunner Framework Risk Adaptive Tiering",
            ),
        ]

    def run_search_benchmark(self) -> List[BenchmarkResult]:
        """Run search benchmark across all strategies and queries."""
        print("=" * 80)
        print("SEARCH BENCHMARK HARNESS")
        print("=" * 80)
        print(f"Strategies: {len(self.strategies)}")
        print(f"Queries: {len(self.queries)}")
        print(f"Top-K: {self.top_k}")
        print(f"Corpus: {self.corpus}")
        print(f"Storage: {self.storage}")
        print("=" * 80)

        for q_label, query in self.queries:
            print(f"\n==================== QUERY {q_label}: {query} ====================")

            for strategy in self.strategies:
                if self.verbose:
                    print(f"--- STRATEGY: {strategy} ---")

                start_time = time.time()

                cmd = [
                    sys.executable,
                    "-m",
                    "src.cli",
                    "search",
                    query,
                    "--strategy",
                    strategy,
                    "--top-k",
                    str(self.top_k),
                    "--corpus",
                    self.corpus,
                    "--storage",
                    self.storage,
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )

                duration_ms = (time.time() - start_time) * 1000

                benchmark_result = BenchmarkResult(
                    query_label=q_label,
                    query=query,
                    strategy=strategy,
                    success=result.returncode == 0,
                    duration_ms=duration_ms,
                    results_count=0,
                    error_message=(result.stderr if result.returncode != 0 else None),
                    output=result.stdout if result.returncode == 0 else None,
                )

                if result.returncode != 0:
                    print(f"ERROR: {result.stderr}")
                    benchmark_result.error_message = result.stderr
                else:
                    # Filter out tqdm / loading weights lines if present, keep search output
                    lines = result.stdout.splitlines()
                    search_lines = []
                    capture = False
                    results_count = 0

                    for line in lines:
                        if line.startswith("Executing search:") or line.startswith("Rank"):
                            capture = True
                        if capture:
                            search_lines.append(line)
                            if line.startswith("#") and line.strip().split()[0].isdigit():
                                results_count += 1

                    benchmark_result.results_count = results_count

                    if search_lines:
                        if self.verbose:
                            print("\n".join(search_lines))
                    else:
                        if self.verbose:
                            print(result.stdout)

                self.results.append(benchmark_result)

                if self.verbose:
                    print(
                        f"Duration: {duration_ms:.2f}ms | Results: {benchmark_result.results_count}"
                    )
                print()

        return self.results

    def run_generation_benchmark(self, live_mode: bool = False) -> List[BenchmarkResult]:
        """Run generation benchmark with different providers using real corpus queries."""
        print("=" * 80)
        print("GENERATION BENCHMARK HARNESS")
        print("=" * 80)
        print(f"Live Mode: {live_mode}")
        print("Using real corpus queries from README")
        print("=" * 80)

        generation_results = []

        # Use the same real queries from README for generation testing
        real_queries = self._get_default_queries()

        for q_label, query in real_queries:
            print(f"\n--- Testing Query {q_label}: {query} ---")

            # Test mock generation with real query
            print("  Testing Mock Generation")
            start_time = time.time()

            cmd = [
                sys.executable,
                "-m",
                "src.cli",
                "ask",
                query,
                "--strategy",
                "rrf_dedup_mmr",
                "--top-k",
                "3",
                "--corpus",
                self.corpus,
                "--storage",
                self.storage,
                "--llm",
                "mock",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            duration_ms = (time.time() - start_time) * 1000

            gen_result = BenchmarkResult(
                query_label=f"gen_mock_{q_label}",
                query=query,
                strategy="mock_generation",
                success=result.returncode == 0,
                duration_ms=duration_ms,
                results_count=1 if result.returncode == 0 else 0,
                error_message=(result.stderr if result.returncode != 0 else None),
                output=result.stdout if result.returncode == 0 else None,
            )

            generation_results.append(gen_result)

            if result.returncode != 0:
                print(f"  ERROR: {result.stderr}")
            else:
                print(f"  ✓ Mock generation completed in {duration_ms:.2f}ms")

            # Test live providers if enabled (using first query only to save API quota)
            if live_mode and q_label == "a":  # Only test live on first query
                providers_to_test = []

                if os.environ.get("ANTHROPIC_API_KEY"):
                    providers_to_test.append(("anthropic", "ANTHROPIC_API_KEY"))
                if os.environ.get("OPENAI_API_KEY"):
                    providers_to_test.append(("openai", "OPENAI_API_KEY"))
                if os.environ.get("GEMINI_API_KEY"):
                    providers_to_test.append(("gemini", "GEMINI_API_KEY"))

                for provider, env_key in providers_to_test:
                    print(f"  Testing {provider.upper()} Generation")
                    start_time = time.time()

                    cmd = [
                        sys.executable,
                        "-m",
                        "src.cli",
                        "ask",
                        query,
                        "--strategy",
                        "rrf_dedup_mmr",
                        "--top-k",
                        "3",
                        "--corpus",
                        self.corpus,
                        "--storage",
                        self.storage,
                        "--provider",
                        provider,
                    ]

                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )

                    duration_ms = (time.time() - start_time) * 1000

                    gen_result = BenchmarkResult(
                        query_label=f"gen_{provider}_{q_label}",
                        query=query,
                        strategy=f"{provider}_generation",
                        success=result.returncode == 0,
                        duration_ms=duration_ms,
                        results_count=1 if result.returncode == 0 else 0,
                        error_message=(result.stderr if result.returncode != 0 else None),
                        output=(result.stdout if result.returncode == 0 else None),
                    )

                    generation_results.append(gen_result)

                    if result.returncode != 0:
                        print(f"  ERROR: {result.stderr}")
                    else:
                        print(f"  ✓ {provider.upper()} generation completed in {duration_ms:.2f}ms")

        if not live_mode:
            print("\n--- Live provider tests skipped (use --live flag) ---")

        self.results.extend(generation_results)
        return generation_results

    def run_evaluation_benchmark(self) -> List[BenchmarkResult]:
        """Run full evaluation benchmark using evaluation harness."""
        print("=" * 80)
        print("EVALUATION BENCHMARK HARNESS")
        print("=" * 80)

        start_time = time.time()

        cmd = [
            sys.executable,
            "run_eval.py",
            "--storage",
            self.storage,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        duration_ms = (time.time() - start_time) * 1000

        eval_result = BenchmarkResult(
            query_label="full_eval",
            query="full_evaluation_benchmark",
            strategy="evaluation_harness",
            success=result.returncode == 0,
            duration_ms=duration_ms,
            results_count=1,
            error_message=result.stderr if result.returncode != 0 else None,
            output=result.stdout if result.returncode == 0 else None,
        )

        self.results.append(eval_result)

        if result.returncode != 0:
            print(f"ERROR: {result.stderr}")
        else:
            print(result.stdout)
            print(f"✓ Evaluation completed in {duration_ms:.2f}ms")

        return [eval_result]

    def print_summary(self):
        """Print comprehensive benchmark summary."""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        if not self.results:
            print("No benchmark results to display.")
            return

        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - successful_tests

        print(f"\nTotal Tests: {total_tests}")
        print(f"Successful: {successful_tests} ({successful_tests/total_tests*100:.1f}%)")
        print(f"Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")

        # Performance summary
        if self.results:
            avg_duration = sum(r.duration_ms for r in self.results) / len(self.results)
            print(f"Average Duration: {avg_duration:.2f}ms")

        # Group by strategy
        by_strategy: Dict[str, List[BenchmarkResult]] = {}
        for result in self.results:
            if result.strategy not in by_strategy:
                by_strategy[result.strategy] = []
            by_strategy[result.strategy].append(result)

        print("\nResults by Strategy:")
        for strategy, strategy_results in by_strategy.items():
            strategy_success = sum(1 for r in strategy_results if r.success)
            strategy_total = len(strategy_results)
            avg_duration = sum(r.duration_ms for r in strategy_results) / strategy_total
            msg = (
                f"  {strategy}: {strategy_success}/{strategy_total} success, "
                f"avg {avg_duration:.2f}ms"
            )
            print(msg)

        # Failed tests details
        if failed_tests > 0:
            print("\nFailed Tests:")
            for result in self.results:
                if not result.success:
                    msg = (
                        f"  ✗ {result.strategy} (Query: {result.query_label}): "
                        f"{result.error_message}"
                    )
                    print(msg)

    def export_results(self, output_file: str, format: str = "json"):
        """Export benchmark results to file."""
        if format == "json":
            results_dict = [
                {
                    "query_label": r.query_label,
                    "query": r.query,
                    "strategy": r.strategy,
                    "success": r.success,
                    "duration_ms": r.duration_ms,
                    "results_count": r.results_count,
                    "error_message": r.error_message,
                    "output": (r.output[:500] if r.output else None),  # Truncate long output
                }
                for r in self.results
            ]

            with open(output_file, "w") as f:
                json.dump(results_dict, f, indent=2)

            print(f"\nResults exported to {output_file}")
        else:
            print(f"Unsupported format: {format}")


def main():
    """Main entry point for benchmark harness."""
    parser = argparse.ArgumentParser(
        description="Enhanced Benchmark Harness for Document Hybrid Search System"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["search", "generation", "evaluation", "all"],
        default="search",
        help="Benchmark mode to run",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live API calls for generation benchmarks",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        help="Specific strategies to test (default: all)",
    )
    parser.add_argument(
        "--queries",
        type=str,
        nargs="+",
        help="Specific queries to test (default: predefined set)",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of results to retrieve")
    parser.add_argument("--corpus", type=str, default="corpus", help="Corpus directory")
    parser.add_argument(
        "--storage",
        type=str,
        default="parquet",
        choices=["memory", "parquet"],
        help="Storage backend",
    )
    parser.add_argument("--output", type=str, help="Output file for results (JSON format)")
    parser.add_argument("--quiet", action="store_true", help="Reduce verbose output")

    args = parser.parse_args()

    # Create harness
    harness = BenchmarkHarness(
        strategies=args.strategies,
        queries=None,  # Use default queries for now
        top_k=args.top_k,
        corpus=args.corpus,
        storage=args.storage,
        verbose=not args.quiet,
    )

    # Run benchmarks based on mode
    if args.mode in ["search", "all"]:
        harness.run_search_benchmark()

    if args.mode in ["generation", "all"]:
        harness.run_generation_benchmark(live_mode=args.live)

    if args.mode in ["evaluation", "all"]:
        harness.run_evaluation_benchmark()

    # Print summary
    harness.print_summary()

    # Export results if requested
    if args.output:
        harness.export_results(args.output, format="json")

    # Return exit code
    return 0 if all(r.success for r in harness.results) else 1


if __name__ == "__main__":
    sys.exit(main())
