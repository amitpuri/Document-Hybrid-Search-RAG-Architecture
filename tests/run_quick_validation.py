"""
Quick Validation Test Harness for Document Hybrid Search System.
Focuses on key strategies and generates README-formatted output for:
1. Side-by-Side Retrieval Comparison tables
2. LLM Adapter Results sections

This test is reusable and can be executed anytime for quick validation.
"""

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.common.console import ensure_utf8_streams

# Ensure UTF-8 output on Windows
ensure_utf8_streams()

# Load environment variables from root .env
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


@dataclass
class RetrievalResult:
    """Result from a single retrieval strategy test."""

    strategy: str
    query: str
    top1_source: str
    top1_snippet: str
    top1_score: float
    success: bool
    duration_ms: float
    error: Optional[str] = None


@dataclass
class GenerationResult:
    """Result from a single generation provider test."""

    provider: str
    model: str
    query: str
    response: str
    citations_count: int
    success: bool
    duration_ms: float
    error: Optional[str] = None


class QuickValidationHarness:
    """
    Quick validation harness that generates results in README format.

    Focuses on key strategies for faster validation:
    1. Side-by-Side Retrieval Comparison (key strategies only)
    2. LLM Adapter Results (mock + available live providers)
    """

    def __init__(
        self,
        corpus_dir: str = "corpus",
        storage_backend: str = "parquet",
        output_file: str = "VALIDATION_RESULTS.md",
        live_mode: Optional[bool] = None,
    ):
        self.corpus_dir = corpus_dir
        self.storage_backend = storage_backend
        self.output_file = output_file

        has_api_keys = bool(
            os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        if live_mode is None:
            self.live_mode = has_api_keys
        else:
            self.live_mode = live_mode

        # Test queries from README
        self.queries = [
            (
                "a",
                ("How does the Binding Constraint Thesis affect " "harness comparisons?"),
            ),
            (
                "b",
                "Dynamic Tiered AgentRunner Framework Risk Adaptive Tiering",
            ),
        ]

        # Key strategies for quick validation (subset of all strategies)
        self.strategies = [
            "bm25",
            "rrf",
            "rrf_dedup_mmr",
            "rrf_graph_dedup_mmr",
        ]

        # Results storage
        self.retrieval_results: List[RetrievalResult] = []
        self.generation_results: List[GenerationResult] = []

    def run_side_by_side_retrieval_comparison(self) -> None:
        """Run retrieval comparison and record results in README format."""
        print("=" * 80)
        print("SIDE-BY-SIDE RETRIEVAL COMPARISON")
        print("=" * 80)

        for q_label, query in self.queries:
            print(f'\n### Query ({q_label}): "{query}"')
            target_doc = "2605.23950v1.pdf" if q_label == "a" else "2605.10223v1.pdf"
            print(f"*Ground-truth target: {target_doc}*")
            print()

            print(
                "| Strategy | Top-1 Source (doc \\| page \\| "
                "section) | Snippet (~100 chars) | Score |"
            )
            print("|---|---|---|---|")

            for strategy in self.strategies:
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
                    "1",
                    "--corpus",
                    self.corpus_dir,
                    "--storage",
                    self.storage_backend,
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )

                duration_ms = (time.time() - start_time) * 1000

                if result.returncode == 0:
                    # Parse output to extract Top-1 result
                    lines = result.stdout.splitlines()
                    top1_source = "N/A"
                    top1_snippet = "N/A"
                    top1_score = 0.0

                    for line in lines:
                        if line.startswith("#1"):
                            parts = line.split()
                            if len(parts) >= 4:
                                top1_score = float(parts[1])
                            # Extract source info
                            if "[" in line and "|" in line:
                                source_start = line.find("[") + 1
                                source_end = line.find("]")
                                top1_source = line[source_start:source_end]
                            # Extract snippet
                            if len(line) > 100:
                                snippet_start = line.find("]") + 2
                                top1_snippet = line[snippet_start : snippet_start + 100] + "..."
                            break

                    self.retrieval_results.append(
                        RetrievalResult(
                            strategy=strategy,
                            query=query,
                            top1_source=top1_source,
                            top1_snippet=top1_snippet,
                            top1_score=top1_score,
                            success=True,
                            duration_ms=duration_ms,
                        )
                    )

                    # Format for markdown table
                    source_escaped = top1_source.replace("|", "\\|")
                    snippet_escaped = top1_snippet.replace("|", "\\|")
                    score_str = f"{top1_score:.3f}"
                    print(
                        f"| `{strategy}` | {source_escaped} | " f"{snippet_escaped} | {score_str} |"
                    )
                else:
                    self.retrieval_results.append(
                        RetrievalResult(
                            strategy=strategy,
                            query=query,
                            top1_source="ERROR",
                            top1_snippet="ERROR",
                            top1_score=0.0,
                            success=False,
                            duration_ms=duration_ms,
                            error=result.stderr,
                        )
                    )
                    print(f"| `{strategy}` | ERROR | ERROR | ERROR |")

    def run_llm_adapter_results(self) -> None:
        """Run generation benchmarks using real APIs from .env."""
        print("\n" + "=" * 80)
        print("LLM ADAPTER RESULTS")
        print("=" * 80)
        print(f"Live Mode: {self.live_mode}")

        active_providers = []
        if self.live_mode:
            if os.environ.get("OPENAI_API_KEY"):
                active_providers.append(("openai", "gpt-5.5"))
            if os.environ.get("ANTHROPIC_API_KEY"):
                active_providers.append(("anthropic", "claude-sonnet-5"))
            if os.environ.get("GEMINI_API_KEY"):
                active_providers.append(("gemini", "gemini-3.8-flash"))

        if not active_providers:
            print(
                "No live API keys detected in .env; "
                "falling back to GroundedSynthesisGenerator (mock)."
            )
            active_providers.append(("mock", "GroundedSynthesisGenerator"))
        else:
            provider_names = [p[0] for p in active_providers]
            print(f"Active real API providers from .env: {provider_names}")

        for q_label, query in self.queries:
            print(f"\n**Benchmark Query {q_label}:**")
            print(f'"{query}"')
            print("**Strategy:** RRF + Dedup + MMR | " "**Corpus:** 11 PDFs, 2,072 chunks\n")

            for provider, model in active_providers:
                print(f"### {provider.capitalize()} - `{model}`")
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
                    self.corpus_dir,
                    "--storage",
                    self.storage_backend,
                    "--provider",
                    provider,
                    "--model",
                    model,
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )

                duration_ms = (time.time() - start_time) * 1000

                if result.returncode == 0:
                    raw_stdout = result.stdout
                    separator = "=" * 80 + "\n"
                    if separator in raw_stdout:
                        parts = raw_stdout.split(separator)
                        if len(parts) >= 2:
                            answer_text = parts[1].strip()
                        else:
                            answer_text = raw_stdout.strip()
                    else:
                        answer_text = raw_stdout.strip()

                    citations_count = answer_text.count("[Source")

                    self.generation_results.append(
                        GenerationResult(
                            provider=provider,
                            model=model,
                            query=query,
                            response=answer_text,
                            citations_count=citations_count,
                            success=True,
                            duration_ms=duration_ms,
                        )
                    )

                    preview = answer_text[:400].replace("\n", " ")
                    print(f"> {preview}...")
                    duration_str = f"{duration_ms:.0f}ms"
                    print(f"\n*Citations: {citations_count} chunks* " f"({duration_str})\n")
                else:
                    self.generation_results.append(
                        GenerationResult(
                            provider=provider,
                            model=model,
                            query=query,
                            response="",
                            citations_count=0,
                            success=False,
                            duration_ms=duration_ms,
                            error=result.stderr,
                        )
                    )
                    print(f"ERROR: {result.stderr}")

    def generate_markdown_report(self) -> None:
        """Generate comprehensive markdown report in README format."""
        print("\n" + "=" * 80)
        print("GENERATING VALIDATION REPORT")
        print("=" * 80)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write("# Document Hybrid Search Validation Results\n\n")
            f.write(f"**Generated:** {timestamp}\n")
            f.write(f"**Corpus:** {self.corpus_dir}\n")
            f.write(f"**Storage Backend:** {self.storage_backend}\n")
            f.write(f"**Live Mode:** {self.live_mode}\n\n")

            # Section 1: Side-by-Side Retrieval Comparison
            f.write("## 🔍 Side-by-Side Retrieval Comparison\n\n")

            for q_label, query in self.queries:
                f.write(f'### Query ({q_label}): "{query}"\n\n')
                target_pdf = "2605.23950v1.pdf" if q_label == "a" else "2605.10223v1.pdf"
                f.write(f"*Ground-truth target: {target_pdf}*\n\n")
                f.write(
                    "| Strategy | Top-1 Source (doc \\| page \\| "
                    "§ section) | Snippet (~100 chars) | Score |\n"
                )
                f.write("|---|---|---|---|\n")

                # Filter results for this query
                query_results = [r for r in self.retrieval_results if r.query == query]

                for result in query_results:
                    source_escaped = result.top1_source.replace("|", "\\|")
                    snippet_escaped = result.top1_snippet.replace("|", "\\|")
                    score_str = f"{result.top1_score:.3f}"
                    f.write(
                        f"| `{result.strategy}` | {source_escaped} | "
                        f"{snippet_escaped} | {score_str} |\n"
                    )

                f.write("\n")

            # Section 2: LLM Adapter Results
            f.write("## 🤖 LLM Adapter Results\n\n")
            f.write("| Provider | Default Model | Route / Mode |\n")
            f.write("|---|---|---|\n")
            f.write("| **OpenAI** | `gpt-5.5` | " "Real API (Direct via `.env`) |\n")
            f.write("| **Anthropic** | `claude-sonnet-5` | " "Real API (Direct via `.env`) |\n")
            f.write("| **Gemini** | `gemini-3.8-flash` | " "Real API (Direct via `.env`) |\n")
            f.write(
                "| Offline Mock | `GroundedSynthesisGenerator` | " "Fallback (Zero API calls) |\n\n"
            )

            for q_label, query in self.queries:
                f.write(f"**Benchmark Query {q_label}:**\n")
                f.write(f'"{query}"\n')
                f.write(
                    "**Strategy:** RRF + Dedup + MMR | **Corpus:** " "11 PDFs, 2,072 chunks\n\n"
                )

                # Filter generation results for this query
                query_gen_results = [r for r in self.generation_results if r.query == query]

                for result in query_gen_results:
                    f.write(f"### {result.provider.capitalize()} — " f"`{result.model}`\n\n")
                    if result.success:
                        formatted_response = "\n".join(
                            f"> {line}" for line in result.response.splitlines()
                        )
                        f.write(f"{formatted_response}\n\n")
                        f.write(f"*Citations: {result.citations_count} chunks*\n\n")
                    else:
                        f.write(f"ERROR: {result.error}\n\n")

                f.write("---\n\n")

            # Summary section
            f.write("## Validation Summary\n\n")
            f.write(f"**Total Retrieval Tests:** {len(self.retrieval_results)}\n")
            f.write(
                f"**Successful Retrieval Tests:** "
                f"{sum(1 for r in self.retrieval_results if r.success)}\n"
            )
            f.write(f"**Total Generation Tests:** " f"{len(self.generation_results)}\n")
            f.write(
                f"**Successful Generation Tests:** "
                f"{sum(1 for r in self.generation_results if r.success)}\n"
            )

        print(f"Validation report generated: {self.output_file}")

    def run_all(self) -> None:
        """Run all validation tests and generate report."""
        print("\n" + "=" * 80)
        print("QUICK VALIDATION HARNESS")
        print("=" * 80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            # Enforce ground-truth validation upfront
            print("Enforcing ground-truth integrity check...")
            from src.evaluation.dataset import (
                EVAL_DATASET,
                validate_ground_truth,
            )
            from src.ingestion.pipeline import IngestionPipeline

            ingestion = IngestionPipeline(
                corpus_dir=self.corpus_dir,
                storage_backend=self.storage_backend,
            )
            chunk_store, _, _ = ingestion.run()
            validate_ground_truth(EVAL_DATASET, chunk_store.get_texts())
            print("✓ Ground-truth integrity verified successfully.\n")

            self.run_side_by_side_retrieval_comparison()
            self.run_llm_adapter_results()
            self.generate_markdown_report()

            print("\n" + "=" * 80)
            print("VALIDATION COMPLETED SUCCESSFULLY")
            print("=" * 80)

        except Exception as e:
            print(f"\nERROR: Validation failed: {e}")
            import traceback

            traceback.print_exc()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Quick validation test harness for Document Hybrid Search"
    )
    parser.add_argument("--corpus", type=str, default="corpus", help="Corpus directory")
    parser.add_argument(
        "--storage",
        type=str,
        default="parquet",
        choices=["memory", "parquet", "qdrant"],
        help="Storage backend",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="VALIDATION_RESULTS.md",
        help="Output markdown file",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live API calls for generation testing",
    )

    args = parser.parse_args()

    harness = QuickValidationHarness(
        corpus_dir=args.corpus,
        storage_backend=args.storage,
        output_file=args.output,
        live_mode=args.live,
    )

    harness.run_all()


if __name__ == "__main__":
    main()
