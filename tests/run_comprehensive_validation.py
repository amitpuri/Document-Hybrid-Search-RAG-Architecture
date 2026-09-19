"""
Comprehensive Validation Test Harness for Document Hybrid Search System.

Generates benchmark results in the same format as README.md sections:
1. Empirical Benchmark Results table (including P0 ablation cells)
2. Side-by-Side Retrieval Comparison tables
3. LLM Adapter Results sections (including C3 confidence tracking)

This test is reusable and can be executed anytime to validate system
performance.

Recent Updates:
- Added P0 ablation cells: graph_only, rrf_graph, rrf_graph_dedup
- Added C3 confidence tracking for generation results (HIGH/MEDIUM/LOW)
"""

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from src.common.console import ensure_utf8_streams
from src.evaluation.harness import EvaluationHarness
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.pipeline import RetrievalPipeline

# Load environment variables from root .env
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# Ensure UTF-8 output on Windows
ensure_utf8_streams()


@dataclass
class ValidationQuery:
    """Benchmark query with ground-truth target and topic context."""

    label: str
    query: str
    target_doc: str
    topic: str = ""


# Curated benchmark queries spanning both legacy core papers and
# the expanded 26-PDF corpus
DEFAULT_VALIDATION_QUERIES: List[ValidationQuery] = [
    # --- Legacy Core Benchmark Queries ---
    ValidationQuery(
        label="a",
        query=(
            "How does the Binding Constraint Thesis affect " "harness comparisons across models?"
        ),
        target_doc="2605.23950v1.pdf",
        topic="Agent Harness Benchmarks & Binding Constraint Thesis",
    ),
    ValidationQuery(
        label="b",
        query="Dynamic Tiered AgentRunner Framework Risk Adaptive Tiering",
        target_doc="2605.10223v1.pdf",
        topic="Enterprise Agent Governance & Risk Adaptive Tiering",
    ),
    # --- Expanded Literature Benchmark Queries ---
    ValidationQuery(
        label="c",
        query=("AlphaGenome regulatory variant effect prediction " "non-coding DNA"),
        target_doc="s41586-025-10014-0.pdf",
        topic="DeepMind AlphaGenome & Regulatory Variant Effect Prediction",
    ),
    ValidationQuery(
        label="d",
        query=("Scalable watermarking for identifying large language " "model outputs SynthID"),
        target_doc="s41586-024-08025-4.pdf",
        topic="LLM Output Provenance & Scalable Watermarking (SynthID)",
    ),
    ValidationQuery(
        label="e",
        query=("Procedural Graphs Self-Evolving Execution Structures " "for LLM Agents"),
        target_doc="2609.09153v1.pdf",
        topic="Self-Evolving Execution Structures & Procedural Graphs",
    ),
    ValidationQuery(
        label="f",
        query=("Analyzing and Predicting Token Consumption in " "Agentic Coding Tasks"),
        target_doc="2604.22750v2.pdf",
        topic="Agentic Coding Economics & Token Consumption",
    ),
    ValidationQuery(
        label="g",
        query=("CTIFOUNDRY AGENT-NATIVE CORPUS SCAFFOLD FOR CYBER " "THREAT INTELLIGENCE"),
        target_doc="2608.18613v1.pdf",
        topic=("Agent-Native Cyber Threat Intelligence & " "Incident Investigation"),
    ),
    ValidationQuery(
        label="h",
        query=("CliniCARE-Bench Clinical Calibrated Audit of Medical " "Reasoning in EHR"),
        target_doc="2608.07796v1.pdf",
        topic=("Clinical Medical Reasoning & Electronic Health Record " "Audit"),
    ),
    ValidationQuery(
        label="i",
        query=("Thinking Fast Slow and Artificial Tri-System Theory " "Cognitive Surrender"),
        target_doc="ssrn-6097646.pdf",
        topic="Cognitive Science & Human-AI Decision Making",
    ),
    ValidationQuery(
        label="j",
        query=("AI Safety Not Optional autonomous agent scaffolds " "and software harness"),
        target_doc="2609.10630v1.pdf",
        topic=("Multi-layer AI Safety Controls & Agent Scaffolds " "(Bengio)"),
    ),
    ValidationQuery(
        label="k",
        query=("Dream-RSI Recursive Self-Improvement through " "Evolving Worlds exploration"),
        target_doc="2609.14858v1.pdf",
        topic="Recursive Self-Improvement & Exploration Simulation",
    ),
    ValidationQuery(
        label="l",
        query=("Artificial intelligence in drug discovery " "translational relevance benchmarking"),
        target_doc="s41573-026-01496-2.pdf",
        topic="AI in Drug Discovery & Translational Benchmarking",
    ),
]


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
    # Added for C3 confidence tracking
    confidence_levels: Optional[List[str]] = None


@dataclass
class StrategyMetrics:
    """Metrics for a single retrieval strategy."""

    strategy: str
    mrr: float
    recall_1: float
    recall_3: float
    recall_5: float
    ndcg_5: float
    entity_coverage: float
    relation_coverage: float


class ComprehensiveValidationHarness:
    """
    Comprehensive validation harness that generates results in README format.

    Produces three main sections:
    1. Empirical Benchmark Results (strategy performance metrics)
    2. Side-by-Side Retrieval Comparison (per-query strategy results)
    3. LLM Adapter Results (generation provider outputs)
    """

    def __init__(
        self,
        corpus_dir: str = "corpus",
        storage_backend: str = "parquet",
        output_file: str = "VALIDATION_RESULTS.md",
        live_mode: Optional[bool] = None,
        query_preset: str = "core",
        query_labels: Optional[List[str]] = None,
        queries: Optional[List[Any]] = None,
        include_ablation: bool = True,
        track_confidence: bool = True,
    ):
        self.corpus_dir = corpus_dir
        self.storage_backend = storage_backend
        self.output_file = output_file
        self.include_ablation = include_ablation
        self.track_confidence = track_confidence

        # Check if live provider keys exist in environment / .env
        has_api_keys = bool(
            os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        # Default to real API if keys exist in .env unless explicitly disabled
        if live_mode is None:
            self.live_mode = has_api_keys
        else:
            self.live_mode = live_mode

        # Configure benchmark queries
        if queries:
            normalized = []
            for item in queries:
                if isinstance(item, ValidationQuery):
                    normalized.append(item)
                elif isinstance(item, (list, tuple)):
                    lbl = str(item[0])
                    q_text = str(item[1])
                    t_doc = str(item[2]) if len(item) > 2 else "Unknown"
                    t_topic = str(item[3]) if len(item) > 3 else ""
                    normalized.append(
                        ValidationQuery(
                            label=lbl,
                            query=q_text,
                            target_doc=t_doc,
                            topic=t_topic,
                        )
                    )
            self.queries = normalized
        elif query_labels:
            requested = {label.strip().lower() for label in query_labels}
            self.queries = [q for q in DEFAULT_VALIDATION_QUERIES if q.label.lower() in requested]
        elif query_preset == "all":
            self.queries = list(DEFAULT_VALIDATION_QUERIES)
        elif query_preset == "new":
            self.queries = [q for q in DEFAULT_VALIDATION_QUERIES if q.label not in ("a", "b")]
        else:  # "core" (default: a, b, c, d)
            self.queries = [
                q for q in DEFAULT_VALIDATION_QUERIES if q.label in ("a", "b", "c", "d")
            ]

        self.corpus_stats_str = f"Corpus: {self.corpus_dir}"

        # All strategies from README + ablation cells (P0)
        self.strategies = [
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
            "rrf_graph_dedup_mmr",
            "qdrant",
        ]

        # Add ablation cells if enabled
        if self.include_ablation:
            self.strategies.extend(
                [
                    "graph_only",
                    "rrf_graph",
                    "rrf_graph_dedup",
                ]
            )

        # Results storage
        self.retrieval_results: List[RetrievalResult] = []
        self.generation_results: List[GenerationResult] = []
        self.strategy_metrics: List[StrategyMetrics] = []

    def run_side_by_side_retrieval_comparison(self) -> None:
        """Run retrieval comparison and record results in README format."""
        print("=" * 80)
        print("SIDE-BY-SIDE RETRIEVAL COMPARISON")
        print("=" * 80)

        for q in self.queries:
            target_desc = f"{q.target_doc}" + (f" ({q.topic})" if q.topic else "")
            print(f'\n### Query ({q.label}): "{q.query}"')
            print(f"*Ground-truth target: {target_desc}*")
            print()

            print(
                "| Strategy | Top-1 Source (doc \\| page \\| § section) | "
                "Snippet (~100 chars) | Score |"
            )
            print("|---|---|---|---|")

            for strategy in self.strategies:
                start_time = time.time()

                cmd = [
                    sys.executable,
                    "-m",
                    "src.cli",
                    "search",
                    q.query,
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
                            query=q.query,
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
                    print(
                        f"| `{strategy}` | {source_escaped} | "
                        f"{snippet_escaped} | {top1_score:.3f} |"
                    )
                else:
                    self.retrieval_results.append(
                        RetrievalResult(
                            strategy=strategy,
                            query=q.query,
                            top1_source="ERROR",
                            top1_snippet="ERROR",
                            top1_score=0.0,
                            success=False,
                            duration_ms=duration_ms,
                            error=result.stderr,
                        )
                    )
                    print(f"| `{strategy}` | ERROR | ERROR | ERROR |")

    def run_empirical_benchmark_evaluation(self) -> None:
        """Run full evaluation and record metrics in README format."""
        print("\n" + "=" * 80)
        print("EMPIRICAL BENCHMARK EVALUATION")
        print("=" * 80)

        print(f"Running evaluation with {self.storage_backend} " "storage backend...")

        ingestion = IngestionPipeline(
            corpus_dir=self.corpus_dir, storage_backend=self.storage_backend
        )
        chunk_store, total_pages, pdf_paths = ingestion.run()
        self.corpus_stats_str = f"{len(pdf_paths)} PDFs, {len(chunk_store):,} chunks"

        print(
            f"Loaded {len(pdf_paths)} PDFs | Total Pages: {total_pages} | "
            f"Chunks: {len(chunk_store)}"
        )

        retrieval = RetrievalPipeline(chunk_store)
        harness = EvaluationHarness(retrieval_pipeline=retrieval)

        print("Running evaluation across all strategies...")
        metrics = harness.run()

        # Convert to StrategyMetrics format
        for strategy_name, strategy_metrics in metrics.items():
            self.strategy_metrics.append(
                StrategyMetrics(
                    strategy=strategy_name,
                    mrr=strategy_metrics["mrr"],
                    recall_1=strategy_metrics["recall_1"],
                    recall_3=strategy_metrics["recall_3"],
                    recall_5=strategy_metrics["recall_5"],
                    ndcg_5=strategy_metrics["ndcg_5"],
                    entity_coverage=strategy_metrics["entity_coverage"],
                    relation_coverage=strategy_metrics["relation_coverage"],
                )
            )

    def run_llm_adapter_results(self) -> None:
        """Run generation benchmarks using real APIs from .env and record
        results."""
        print("\n" + "=" * 80)
        print("LLM ADAPTER RESULTS")
        print("=" * 80)
        print(f"Live Mode: {self.live_mode}")

        # Detect active providers from .env
        active_providers = []
        if self.live_mode:
            if os.environ.get("OPENAI_API_KEY"):
                active_providers.append(("openai", "gpt-5.5"))
            if os.environ.get("ANTHROPIC_API_KEY"):
                active_providers.append(("anthropic", "claude-sonnet-5"))
            if os.environ.get("GEMINI_API_KEY"):
                active_providers.append(("gemini", "gemini-3.8-flash"))

        # Fallback to mock if no live keys are found or mode is disabled
        if not active_providers:
            print(
                "No live API keys detected in .env; falling back to "
                "GroundedSynthesisGenerator (mock)."
            )
            active_providers.append(("mock", "GroundedSynthesisGenerator"))
        else:
            provider_names = [p[0] for p in active_providers]
            provider_msg = f"Active real API providers from .env: {provider_names}"
            print(provider_msg)

        for q in self.queries:
            print(f"\n**Benchmark Query {q.label}:**")
            print(f'"{q.query}"')
            strategy_msg = (
                f"**Strategy:** RRF + Dedup + MMR | " f"**Corpus:** {self.corpus_stats_str}\n"
            )
            print(strategy_msg)

            for provider, model in active_providers:
                print(f"### {provider.capitalize()} — `{model}`")
                start_time = time.time()

                cmd = [
                    sys.executable,
                    "-m",
                    "src.cli",
                    "ask",
                    q.query,
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
                    # Extract answer between separator banners if present
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

                    # Extract confidence levels (C3 feature)
                    confidence_levels = []
                    for line in answer_text.splitlines():
                        if "[HIGH]" in line:
                            confidence_levels.append("HIGH")
                        elif "[MED]" in line or "[MEDIUM]" in line:
                            confidence_levels.append("MEDIUM")
                        elif "[LOW]" in line:
                            confidence_levels.append("LOW")

                    self.generation_results.append(
                        GenerationResult(
                            provider=provider,
                            model=model,
                            query=q.query,
                            response=answer_text,
                            citations_count=citations_count,
                            success=True,
                            duration_ms=duration_ms,
                            confidence_levels=(confidence_levels if confidence_levels else None),
                        )
                    )

                    preview = answer_text[:400].replace("\n", " ")
                    print(f"> {preview}...")
                    citations_str = (
                        f"*Citations: {citations_count} chunks* " f"({duration_ms:.0f}ms)"
                    )
                    print(f"\n{citations_str}")
                    if confidence_levels:
                        high_count = confidence_levels.count("HIGH")
                        med_count = confidence_levels.count("MEDIUM")
                        low_count = confidence_levels.count("LOW")
                        conf_msg = (
                            f"*Confidence: {high_count} HIGH, "
                            f"{med_count} MEDIUM, {low_count} LOW*\n"
                        )
                        print(conf_msg)
                    else:
                        print()
                else:
                    self.generation_results.append(
                        GenerationResult(
                            provider=provider,
                            model=model,
                            query=q.query,
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

            # Section 1: Empirical Benchmark Results
            f.write("## 📊 Empirical Benchmark Results\n\n")
            f.write(
                "| # | Strategy Name | MRR | Recall@1 | Recall@3 | "
                "Recall@5 | NDCG@5 | Key Characteristic |\n"
            )
            f.write("|---|---|---|---|---|---|---|---|\n")

            for i, metrics in enumerate(self.strategy_metrics, 1):
                row = (
                    f"| {i} | {metrics.strategy} | {metrics.mrr:.3f} | "
                    f"{metrics.recall_1:.3f} | {metrics.recall_3:.3f} | "
                    f"{metrics.recall_5:.3f} | {metrics.ndcg_5:.3f} | "
                )
                f.write(row)

                # Add characteristic based on strategy
                if "bm25" in metrics.strategy:
                    f.write("Exceptional keyword precision on domain jargon |\n")
                elif "tfidf" in metrics.strategy:
                    f.write("Vector space baseline with sublinear term " "frequencies |\n")
                elif "linear" in metrics.strategy:
                    f.write("Linear score combination |\n")
                elif (
                    "rrf" in metrics.strategy
                    and "dedup" not in metrics.strategy
                    and "mmr" not in metrics.strategy
                    and "graph" not in metrics.strategy
                ):
                    f.write("Immune to score-scale distortion |\n")
                elif "rrf_dedup" in metrics.strategy:
                    f.write("Eliminates redundant sliding-window chunk " "overlap |\n")
                elif "rrf_dedup_mmr" in metrics.strategy:
                    f.write("Top ranking diversity via MMR (lambda=0.7) |\n")
                elif "rrf_graph" in metrics.strategy:
                    f.write("Fuses IDF-weighted NetworkX KG into RRF |\n")
                elif "ppmi" in metrics.strategy:
                    f.write("Zero-dependency distributional semantics " "from scratch |\n")
                elif "cross_encoder" in metrics.strategy:
                    f.write("Re-ranks 50 un-deduplicated candidates via " "ms-marco |\n")
                elif "sentence_transformer" in metrics.strategy:
                    f.write("Pure dense bi-encoder; diffuses rare coined " "terms |\n")
                elif "adaptive" in metrics.strategy:
                    f.write("Dynamic query-intent alpha weighting " "heuristic |\n")
                elif "specter2" in metrics.strategy:
                    f.write("Domain-adapted scientific embedding |\n")
                elif "qdrant" in metrics.strategy.lower():
                    f.write(
                        "High-speed approximate nearest neighbor (ANN) "
                        "vector retrieval via Qdrant HNSW index |\n"
                    )
                elif "graph_only" in metrics.strategy:
                    f.write(
                        "Ablation: Graph retrieval alone (no BM25/TF-IDF, " "no postprocessing) |\n"
                    )
                elif "rrf_graph" in metrics.strategy and "dedup" not in metrics.strategy:
                    f.write("Ablation: RRF fusion of BM25 + TF-IDF + Graph " "(no dedup/MMR) |\n")
                elif "rrf_graph_dedup" in metrics.strategy and "mmr" not in metrics.strategy:
                    f.write("Ablation: RRF + Graph + Dedup (no MMR) |\n")
                else:
                    f.write("|\n")

            # Section 2: Side-by-Side Retrieval Comparison
            f.write("\n## 🔍 Side-by-Side Retrieval Comparison\n\n")

            for q in self.queries:
                target_desc = f"{q.target_doc}" + (f" ({q.topic})" if q.topic else "")
                f.write(f'### Query ({q.label}): "{q.query}"\n\n')
                f.write(f"*Ground-truth target: {target_desc}*\n\n")
                f.write(
                    "| Strategy | Top-1 Source (doc \\| page \\| § section) | "
                    "Snippet (~100 chars) | Score |\n"
                )
                f.write("|---|---|---|---|\n")

                # Filter results for this query
                query_results = [r for r in self.retrieval_results if r.query == q.query]

                for result in query_results:
                    source_escaped = result.top1_source.replace("|", "\\|")
                    snippet_escaped = result.top1_snippet.replace("|", "\\|")
                    f.write(
                        f"| `{result.strategy}` | {source_escaped} | "
                        f"{snippet_escaped} | {result.top1_score:.3f} |\n"
                    )

                f.write("\n")

            # Section 3: LLM Adapter Results
            f.write("## 🤖 LLM Adapter Results\n\n")
            f.write("| Provider | Default Model | Route / Mode |\n")
            f.write("|---|---|---|\n")
            f.write("| **OpenAI** | `gpt-5.5` | Real API (Direct via `.env`) |\n")
            f.write("| **Anthropic** | `claude-sonnet-5` | " "Real API (Direct via `.env`) |\n")
            f.write("| **Gemini** | `gemini-3.8-flash` | " "Real API (Direct via `.env`) |\n")
            f.write(
                "| Offline Mock | `GroundedSynthesisGenerator` | " "Fallback (Zero API calls) |\n\n"
            )

            for q in self.queries:
                f.write(f"**Benchmark Query {q.label}:**\n")
                f.write(f'"{q.query}"\n')
                f.write(
                    f"**Strategy:** RRF + Dedup + MMR | " f"**Corpus:** {self.corpus_stats_str}\n\n"
                )

                # Filter generation results for this query
                query_gen_results = [r for r in self.generation_results if r.query == q.query]

                for result in query_gen_results:
                    model_header = f"### {result.provider.capitalize()} — " f"`{result.model}`\n\n"
                    f.write(model_header)
                    if result.success:
                        formatted_response = "\n".join(
                            f"> {line}" for line in result.response.splitlines()
                        )
                        f.write(f"{formatted_response}\n\n")
                        citations_msg = f"*Citations: {result.citations_count} chunks*\n"
                        f.write(citations_msg)
                        if result.confidence_levels:
                            high_count = result.confidence_levels.count("HIGH")
                            med_count = result.confidence_levels.count("MEDIUM")
                            low_count = result.confidence_levels.count("LOW")
                            confidence_str = (
                                f"*Confidence: {high_count} HIGH, "
                                f"{med_count} MEDIUM, {low_count} LOW*\n"
                            )
                            f.write(confidence_str)
                        f.write("\n")
                    else:
                        f.write(f"ERROR: {result.error}\n\n")

                f.write("---\n\n")

            # Summary section
            f.write("## Validation Summary\n\n")
            total_msg = f"**Total Retrieval Tests:** " f"{len(self.retrieval_results)}\n"
            f.write(total_msg)
            f.write(
                f"**Successful Retrieval Tests:** "
                f"{sum(1 for r in self.retrieval_results if r.success)}\n"
            )
            gen_total_msg = f"**Total Generation Tests:** " f"{len(self.generation_results)}\n"
            f.write(gen_total_msg)
            f.write(
                f"**Successful Generation Tests:** "
                f"{sum(1 for r in self.generation_results if r.success)}\n"
            )

        print(f"Validation report generated: {self.output_file}")

    def run_all(self) -> None:
        """Run all validation tests and generate report."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE VALIDATION HARNESS")
        print("=" * 80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            # Enforce ground-truth validation upfront
            print("Enforcing ground-truth integrity check...")
            from src.evaluation.dataset import (
                EVAL_DATASET,
                validate_ground_truth,
            )
            from src.ingestion.pipeline import (
                IngestionPipeline,
            )

            ingestion = IngestionPipeline(
                corpus_dir=self.corpus_dir,
                storage_backend=self.storage_backend,
            )
            chunk_store, _, pdf_paths = ingestion.run()
            self.corpus_stats_str = f"{len(pdf_paths)} PDFs, {len(chunk_store):,} chunks"
            validate_ground_truth(EVAL_DATASET, chunk_store.get_texts())
            msg = f"✓ Ground-truth integrity verified successfully " f"({self.corpus_stats_str}).\n"
            print(msg)

            self.run_side_by_side_retrieval_comparison()
            self.run_empirical_benchmark_evaluation()
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
        description=("Comprehensive validation test harness for " "Document Hybrid Search")
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
        dest="live",
        action="store_true",
        default=None,
        help=("Enable live API calls for generation testing " "(auto-detected if .env has keys)"),
    )
    parser.add_argument(
        "--no-live",
        dest="live",
        action="store_false",
        help="Force offline mock generation even if .env has keys",
    )
    parser.add_argument(
        "--query-preset",
        type=str,
        default="core",
        choices=["core", "new", "all"],
        help=(
            "Query preset to run in side-by-side and LLM validation: "
            "'core' (default a-d), 'new' (c-l), 'all' (a-l full suite)"
        ),
    )
    parser.add_argument(
        "--query-labels",
        type=str,
        default=None,
        help="Comma-separated query labels to evaluate (e.g. 'a,b,c,e')",
    )
    parser.add_argument(
        "--no-ablation",
        dest="include_ablation",
        action="store_false",
        help=("Exclude P0 ablation cells " "(graph_only, rrf_graph, rrf_graph_dedup)"),
    )
    parser.add_argument(
        "--no-confidence",
        dest="track_confidence",
        action="store_false",
        help="Disable C3 confidence tracking in generation results",
    )

    args = parser.parse_args()

    labels = [lbl.strip() for lbl in args.query_labels.split(",")] if args.query_labels else None

    harness = ComprehensiveValidationHarness(
        corpus_dir=args.corpus,
        storage_backend=args.storage,
        output_file=args.output,
        live_mode=args.live,
        query_preset=args.query_preset,
        query_labels=labels,
        include_ablation=args.include_ablation,
        track_confidence=args.track_confidence,
    )

    harness.run_all()


if __name__ == "__main__":
    main()
