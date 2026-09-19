"""
Command-Line Interface for Hybrid Search, Ingestion, Evaluation, and RAG.
"""

import argparse
import json

from src.common.console import ensure_utf8_streams
from src.config import DEFAULT_STORAGE_BACKEND
from src.engine import HybridSearchEngine
from src.evaluation.harness import EvaluationHarness
from src.ingestion.pipeline import IngestionPipeline

# Ensure UTF-8 output on Windows
ensure_utf8_streams()


def handle_search(args):
    print(f"Loading engine (corpus: {args.corpus}, storage: {args.storage})...")
    engine = HybridSearchEngine.from_corpus(corpus_dir=args.corpus, storage_backend=args.storage)
    if hasattr(args, "graph_mode") and hasattr(engine.retrieval, "graph_retriever"):
        if args.graph_mode == "local":
            engine.retrieval.graph_retriever.local_weight = 1.0
            engine.retrieval.graph_retriever.global_weight = 0.0
        elif args.graph_mode == "global":
            engine.retrieval.graph_retriever.local_weight = 0.0
            engine.retrieval.graph_retriever.global_weight = 1.0

    print(f"Executing search: {args.query!r} " f"[Strategy: {args.strategy} | Top-{args.top_k}]\n")

    results = engine.search(query=args.query, strategy=args.strategy, top_k=args.top_k)

    print(f"{'Rank':<6}{'Score':<8}Excerpt / Match Provenance")
    print("=" * 95)
    for r in results:
        c = r.chunk
        header = f"[{c.doc_name} | Page {c.page_num} | § {c.section}]"
        text_preview = c.text.replace("\n", " ")
        if len(text_preview) > 100:
            text_preview = text_preview[:97] + "..."
        print(f"#{r.rank:<5}{r.score:<8.3f}{header}\n       {text_preview}\n")


def handle_ask(args):
    print(
        f"Loading engine (corpus: {args.corpus}, " f"storage: {args.storage}, llm: {args.llm})..."
    )  # noqa: E501

    # Determine generator configuration
    generator_config = None
    if args.provider and args.provider != "mock":
        # Use specific provider with optional route
        generator_config = {
            "provider": args.provider,
            "route": args.route if args.route else "direct",
            "model": getattr(args, "model", None),
        }
    elif args.router_config:
        # Use router config from file
        with open(args.router_config, "r") as f:
            router_config = json.load(f)
        generator_config = {"router_config": router_config}
    elif args.llm == "mock":
        # Explicitly use mock
        generator_config = {"provider": "mock"}
    else:
        # Default to mock (existing behavior)
        generator_config = {"provider": "mock"}

    engine = HybridSearchEngine.from_corpus(
        corpus_dir=args.corpus,
        storage_backend=args.storage,
        generator_config=generator_config,
    )
    if hasattr(args, "graph_mode") and hasattr(engine.retrieval, "graph_retriever"):
        if args.graph_mode == "local":
            engine.retrieval.graph_retriever.local_weight = 1.0
            engine.retrieval.graph_retriever.global_weight = 0.0
        elif args.graph_mode == "global":
            engine.retrieval.graph_retriever.local_weight = 0.0
            engine.retrieval.graph_retriever.global_weight = 1.0

    print(f"Generating grounded answer for: {args.question!r}\n")

    gen_result = engine.generate_answer(
        query=args.question, strategy=args.strategy, top_k=args.top_k
    )

    print("=" * 80)
    print(gen_result.answer)
    print("=" * 80)

    # Display citation confidence levels if available
    if gen_result.citation_confidence and len(gen_result.citation_confidence) == len(
        gen_result.citations
    ):
        print("\nCitation Confidence Levels:")
        for i, (chunk, confidence) in enumerate(
            zip(gen_result.citations, gen_result.citation_confidence),
            start=1,
        ):
            if confidence == "high":
                confidence_symbol = "[HIGH]"
            elif confidence == "medium":
                confidence_symbol = "[MED]"
            else:
                confidence_symbol = "[LOW]"
            print(
                f"  [{i}] {confidence_symbol} {confidence.upper()}: "
                f"{chunk.doc_name} (Page {chunk.page_num}, § {chunk.section})"
            )


def handle_eval(args):
    print(f"Running evaluation benchmark harness (storage: {args.storage})...")
    ingestion = IngestionPipeline(storage_backend=args.storage)
    chunk_store, total_pages, pdf_paths = ingestion.run()
    from src.retrieval.pipeline import RetrievalPipeline

    pipeline = RetrievalPipeline(chunk_store)
    harness = EvaluationHarness(retrieval_pipeline=pipeline)
    harness.run()


def handle_ingest(args):
    if args.arxiv:
        # arXiv ingestion mode
        print(
            f"Running arXiv Ingestion Pipeline "
            f"(category={args.category}, limit={args.limit})..."
        )
        from src.ingestion.arxiv_fetcher import (
            ArxivFetcher,
            ArxivRateLimiter,
        )

        rate_limiter = ArxivRateLimiter(min_delay=3.0, max_retries=5)
        fetcher = ArxivFetcher(
            categories=([args.category] if args.category else ["cs.AI"]),
            rate_limiter=rate_limiter,
        )

        # Search for papers
        papers = fetcher.search_papers(
            query=args.arxiv_query if args.arxiv_query else "",
            max_results=args.limit,
            sort_by="submittedDate",
        )

        if not papers:
            print("No papers found matching criteria.")
            return

        print(f"Found {len(papers)} papers. Downloading and ingesting...")

        # Initialize ingestion pipeline
        ingestion = IngestionPipeline(corpus_dir=args.corpus, storage_backend=args.storage)

        # Ingest papers
        arxiv_ids = [p.arxiv_id for p in papers]
        ingested_count, failed_ids = fetcher.ingest_to_corpus(
            arxiv_ids=arxiv_ids,
            corpus_dir=args.corpus,
            pipeline=ingestion,
            batch_size=10,
        )

        print("\narXiv Ingestion Complete!")
        print(f"Papers Attempted: {len(arxiv_ids)}")
        print(f"Papers Successfully Ingested: {ingested_count}")
        if failed_ids:
            print(f"Failed Papers: {len(failed_ids)}")
            for fid in failed_ids:
                print(f"  - {fid}")
    else:
        # Standard corpus ingestion
        print(
            f"Running Ingestion Pipeline on: {args.corpus} "
            f"(storage={args.storage}, force_rebuild={args.force})..."
        )
        ingestion = IngestionPipeline(corpus_dir=args.corpus, storage_backend=args.storage)
        chunk_store, total_pages, pdf_paths = ingestion.run(force_rebuild=args.force)
        print("\nIngestion Complete!")
        print(f"Storage Backend: {args.storage}")
        print(f"Total PDFs Ingested: {len(pdf_paths)}")
        print(f"Total Pages Extracted: {total_pages}")
        print(f"Total Structured Chunks: {len(chunk_store)}")


def handle_qdrant_status(args):
    from src.config import QDRANT_COLLECTION_NAME, QDRANT_HOST, QDRANT_PORT

    host = args.host if hasattr(args, "host") and args.host else QDRANT_HOST
    port = args.port if hasattr(args, "port") and args.port else QDRANT_PORT
    print(f"Connecting to Qdrant at {host}:{port}...")
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(host=host, port=port, timeout=5.0)
        collections = client.get_collections()
        col_names = [c.name for c in collections.collections]
        print("✓ Connected to local Qdrant instance successfully!")
        print(f"  Web Dashboard: http://{host}:{port}/dashboard")
        print(f"  Available Collections: {col_names}")
        if QDRANT_COLLECTION_NAME in col_names:
            info = client.get_collection(QDRANT_COLLECTION_NAME)
            print(f"  Collection '{QDRANT_COLLECTION_NAME}':")
            print(f"    - Points count: {info.points_count}")
            print(f"    - Status: {info.status}")
        else:
            print(
                f"  Target collection '{QDRANT_COLLECTION_NAME}' not yet "
                "created. Run 'python -m src.cli ingest --storage qdrant' "
                "to ingest."
            )
    except Exception as e:
        print(f"✗ Failed to connect to Qdrant: {e}")
        print("  Tip: Make sure the local Docker container is running:")
        print("       docker compose up -d")
        print(
            "       or: docker run -d -p 6333:6333 -p 6334:6334 "
            "-v ${PWD}/qdrant_storage:/qdrant/storage:z "
            "--name qdrant_local qdrant/qdrant"
        )


def main():
    parser = argparse.ArgumentParser(description="Document Hybrid Search & Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # eval
    p_eval = subparsers.add_parser(
        "eval",
        help="Run 14-query x 14-strategy evaluation benchmark",
    )
    p_eval.add_argument(
        "--storage",
        type=str,
        default=DEFAULT_STORAGE_BACKEND,
        choices=["parquet", "memory", "qdrant"],
        help="Storage backend",
    )

    # search
    p_search = subparsers.add_parser("search", help="Execute multi-strategy document retrieval")
    p_search.add_argument("query", type=str, help="Search query string")
    p_search.add_argument(
        "--strategy",
        type=str,
        default="rrf_dedup_mmr",
        help="Retrieval strategy name or alias",
    )
    p_search.add_argument(
        "--graph-mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "local", "global"],
        help="Graph retrieval mode",
    )
    p_search.add_argument("--top-k", type=int, default=5, help="Number of results to retrieve")
    p_search.add_argument("--corpus", type=str, default="corpus", help="Corpus directory path")
    p_search.add_argument(
        "--storage",
        type=str,
        default=DEFAULT_STORAGE_BACKEND,
        choices=["parquet", "memory", "qdrant"],
        help="Storage backend",
    )

    # ask (RAG)
    p_ask = subparsers.add_parser(
        "ask",
        help="Ask a question and generate a grounded answer with citations",
    )
    p_ask.add_argument("question", type=str, help="Question string")
    p_ask.add_argument(
        "--strategy",
        type=str,
        default="rrf_graph_dedup_mmr",
        help="Retrieval strategy name or alias",
    )
    p_ask.add_argument(
        "--graph-mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "local", "global"],
        help="Graph retrieval mode",
    )
    p_ask.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieved context passages",
    )
    p_ask.add_argument("--corpus", type=str, default="corpus", help="Corpus directory path")
    p_ask.add_argument(
        "--storage",
        type=str,
        default=DEFAULT_STORAGE_BACKEND,
        choices=["parquet", "memory", "qdrant"],
        help="Storage backend",
    )
    p_ask.add_argument(
        "--llm",
        type=str,
        default="auto",
        choices=["auto", "openai", "gemini", "anthropic", "mock"],
        help=(
            "LLM backend for answer generation. "
            "'auto' detects from env vars "
            "(ANTHROPIC_API_KEY > OPENAI_API_KEY > GEMINI_API_KEY); "
            "'mock' uses the offline synthesizer regardless of env vars."
        ),
    )
    p_ask.add_argument(
        "--provider",
        type=str,
        choices=["anthropic", "openai", "gemini", "mock"],
        help=(
            "LLM provider to use for generation. "
            "If not specified, defaults to 'mock' (offline mode). "
            "Use with --route to specify deployment route."
        ),
    )
    p_ask.add_argument(
        "--route",
        type=str,
        choices=["direct", "bedrock", "azure", "vertex"],
        help=(
            "Deployment route for the provider. "
            "Anthropic: 'direct' (api.anthropic.com) or 'bedrock' (AWS). "
            "OpenAI: 'direct' (api.openai.com) or 'azure' "
            "(Azure OpenAI). "
            "Gemini: 'direct' (generativelanguage.googleapis.com) or "
            "'vertex' (GCP Vertex AI). "
            "Defaults to 'direct' if not specified."
        ),
    )
    p_ask.add_argument(
        "--router-config",
        type=str,
        help=(
            "Path to JSON file containing fallback chain configuration. "
            'Format: [["anthropic", "direct"], '
            '["anthropic", "bedrock"], ["openai", "direct"]]. '
            "If specified, --provider and --route are ignored."
        ),
    )
    p_ask.add_argument(
        "--model",
        type=str,
        help=("Model identifier override (e.g. 'gpt-5', " "'claude-sonnet-5', 'gemini-3.8-flash')"),
    )

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Run ingestion pipeline on corpus directory")
    p_ingest.add_argument("--corpus", type=str, default="corpus", help="Corpus directory path")
    p_ingest.add_argument(
        "--storage",
        type=str,
        default=DEFAULT_STORAGE_BACKEND,
        choices=["parquet", "memory", "qdrant"],
        help="Storage backend",
    )
    p_ingest.add_argument("--force", action="store_true", help="Force rebuild cache")
    # arXiv ingestion flags (Phase 4)
    p_ingest.add_argument("--arxiv", action="store_true", help="Enable arXiv ingestion mode")
    p_ingest.add_argument(
        "--category",
        type=str,
        default="cs.AI",
        help="arXiv category filter (e.g., cs.AI, cs.IR, cs.CL)",
    )
    p_ingest.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of arXiv papers to ingest",
    )
    p_ingest.add_argument(
        "--arxiv-query",
        type=str,
        default="",
        help="Optional search query within arXiv category",
    )

    # qdrant-status
    p_qd = subparsers.add_parser(
        "qdrant-status",
        help="Check local Qdrant vector database health and stats",
    )
    p_qd.add_argument(
        "--host",
        type=str,
        default=None,
        help="Qdrant host (default from config: localhost)",
    )
    p_qd.add_argument(
        "--port",
        type=int,
        default=None,
        help="Qdrant port (default from config: 6333)",
    )

    args = parser.parse_args()

    if args.command == "eval":
        handle_eval(args)
    elif args.command == "search":
        handle_search(args)
    elif args.command == "ask":
        handle_ask(args)
    elif args.command == "ingest":
        handle_ingest(args)
    elif args.command == "qdrant-status":
        handle_qdrant_status(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
