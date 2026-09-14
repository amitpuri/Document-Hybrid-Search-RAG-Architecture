"""
Command-Line Interface for Hybrid Search, Ingestion, Evaluation, and RAG.
"""

import sys
import argparse
from pathlib import Path

# UTF-8 stdout configuration for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.engine import HybridSearchEngine
from src.ingestion.pipeline import IngestionPipeline
from src.evaluation.harness import EvaluationHarness
from src.config import DEFAULT_STORAGE_BACKEND


def handle_search(args):
    print(f"Loading engine (corpus: {args.corpus}, storage: {args.storage})...")
    engine = HybridSearchEngine.from_corpus(corpus_dir=args.corpus, storage_backend=args.storage)
    print(f"Executing search: {args.query!r} [Strategy: {args.strategy} | Top-{args.top_k}]\n")

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
    print(f"Loading engine (corpus: {args.corpus}, storage: {args.storage}, llm: {args.llm})...")
    engine = HybridSearchEngine.from_corpus(
        corpus_dir=args.corpus,
        llm=None if args.llm == "auto" else args.llm,
        storage_backend=args.storage,
    )
    print(f"Generating grounded answer for: {args.question!r}\n")

    gen_result = engine.generate_answer(
        query=args.question,
        strategy=args.strategy,
        top_k=args.top_k
    )

    print("=" * 80)
    print(gen_result.answer)
    print("=" * 80)


def handle_eval(args):
    print(f"Running evaluation benchmark harness (storage: {args.storage})...")
    ingestion = IngestionPipeline(storage_backend=args.storage)
    chunk_store, total_pages, pdf_paths = ingestion.run()
    from src.retrieval.pipeline import RetrievalPipeline
    pipeline = RetrievalPipeline(chunk_store)
    harness = EvaluationHarness(retrieval_pipeline=pipeline)
    harness.run()


def handle_ingest(args):
    print(f"Running Ingestion Pipeline on: {args.corpus} (storage={args.storage}, force_rebuild={args.force})...")
    ingestion = IngestionPipeline(corpus_dir=args.corpus, storage_backend=args.storage)
    chunk_store, total_pages, pdf_paths = ingestion.run(force_rebuild=args.force)
    print(f"\nIngestion Complete!")
    print(f"Storage Backend: {args.storage}")
    print(f"Total PDFs Ingested: {len(pdf_paths)}")
    print(f"Total Pages Extracted: {total_pages}")
    print(f"Total Structured Chunks: {len(chunk_store)}")


def main():
    parser = argparse.ArgumentParser(description="Document Hybrid Search & Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # eval
    p_eval = subparsers.add_parser("eval", help="Run 14-query x 13-strategy evaluation benchmark")
    p_eval.add_argument("--storage", type=str, default=DEFAULT_STORAGE_BACKEND, choices=["parquet", "memory"], help="Storage backend")

    # search
    p_search = subparsers.add_parser("search", help="Search the corpus")
    p_search.add_argument("query", type=str, help="Search query string")
    p_search.add_argument("--strategy", type=str, default="rrf_dedup_mmr", help="Retrieval strategy name or alias")
    p_search.add_argument("--top-k", type=int, default=5, help="Number of results to retrieve")
    p_search.add_argument("--corpus", type=str, default="corpus", help="Corpus directory path")
    p_search.add_argument("--storage", type=str, default=DEFAULT_STORAGE_BACKEND, choices=["parquet", "memory"], help="Storage backend")

    # ask (RAG)
    p_ask = subparsers.add_parser("ask", help="Ask a question and generate a grounded answer with citations")
    p_ask.add_argument("question", type=str, help="Question string")
    p_ask.add_argument("--strategy", type=str, default="rrf_dedup_mmr", help="Retrieval strategy name or alias")
    p_ask.add_argument("--top-k", type=int, default=3, help="Number of retrieved context passages")
    p_ask.add_argument("--corpus", type=str, default="corpus", help="Corpus directory path")
    p_ask.add_argument("--storage", type=str, default=DEFAULT_STORAGE_BACKEND, choices=["parquet", "memory"], help="Storage backend")
    p_ask.add_argument(
        "--llm",
        type=str,
        default="auto",
        choices=["auto", "openai", "gemini", "anthropic", "mock"],
        help=(
            "LLM backend for answer generation. "
            "'auto' detects from env vars (ANTHROPIC_API_KEY > OPENAI_API_KEY > GEMINI_API_KEY); "
            "'mock' uses the offline synthesizer regardless of env vars."
        )
    )

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Run ingestion pipeline on corpus directory")
    p_ingest.add_argument("--corpus", type=str, default="corpus", help="Corpus directory path")
    p_ingest.add_argument("--storage", type=str, default=DEFAULT_STORAGE_BACKEND, choices=["parquet", "memory"], help="Storage backend")
    p_ingest.add_argument("--force", action="store_true", help="Force rebuild cache")

    args = parser.parse_args()

    if args.command == "eval":
        handle_eval(args)
    elif args.command == "search":
        handle_search(args)
    elif args.command == "ask":
        handle_ask(args)
    elif args.command == "ingest":
        handle_ingest(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
