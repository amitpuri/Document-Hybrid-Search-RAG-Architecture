"""
MCP Server for arXiv Corpus Management.

This server provides conversational agent access to arXiv paper discovery
and ingestion capabilities. It is a thin wrapper around src.ingestion.arxiv_fetcher
to ensure Phase 4 contract compliance (rate limiting, append-only chunk IDs).

Tools:
- search_arxiv: Search arXiv for papers by category and query
- fetch_arxiv_paper: Get metadata for a specific arXiv paper
- ingest_to_corpus: Download and ingest arXiv papers into the corpus

Usage:
    python -m src.mcp.arxiv_server
    or
    mcp-server dev src/mcp/arxiv_server.py
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# MCP imports
try:
    from mcp.server import Server
    from mcp.types import TextContent, Tool

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    Server = None
    Tool = None
    TextContent = None

from src.ingestion.arxiv_fetcher import (
    ArxivFetcher,
    ArxivFetchError,
    ArxivRateLimiter,
)

logger = logging.getLogger(__name__)

# Global fetcher instance (lazy initialization)
_fetcher: Optional[ArxivFetcher] = None


def get_fetcher() -> ArxivFetcher:
    """Get or create the global ArxivFetcher instance."""
    global _fetcher
    if _fetcher is None:
        rate_limiter = ArxivRateLimiter(min_delay=3.0, max_retries=5)
        _fetcher = ArxivFetcher(categories=["cs.AI", "cs.IR", "cs.CL"], rate_limiter=rate_limiter)
    return _fetcher


def search_arxiv(
    query: str,
    category: str = "cs.AI",
    max_results: int = 20,
    date_from: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search arXiv for papers matching category and query criteria.

    Args:
        query: Search query string (e.g., "reinforcement learning")
        category: arXiv category filter (e.g., cs.AI, cs.IR, cs.CL)
        max_results: Maximum number of results to return (default: 20)
        date_from: Optional date filter in YYYY-MM-DD format

    Returns:
        List of paper metadata dictionaries containing:
        - arxiv_id: arXiv identifier
        - title: Paper title
        - authors: List of author names
        - abstract: Paper abstract
        - published: Publication date
        - categories: List of arXiv categories
        - pdf_url: Direct PDF download URL
        - primary_category: Primary arXiv category
    """
    fetcher = get_fetcher()
    fetcher.categories = [category]

    try:
        papers = fetcher.search_papers(
            query=query,
            max_results=max_results,
            date_from=date_from,
            sort_by="submittedDate",
        )
        return [paper.to_dict() for paper in papers]
    except ArxivFetchError as e:
        logger.error(f"arXiv search failed: {e}")
        raise


def fetch_arxiv_paper(arxiv_id: str) -> Optional[Dict[str, Any]]:
    """
    Get metadata for a specific arXiv paper without downloading PDF.

    Args:
        arxiv_id: arXiv identifier (e.g., "2301.07041")

    Returns:
        Paper metadata dictionary or None if not found
    """
    fetcher = get_fetcher()

    try:
        paper = fetcher.get_paper_metadata(arxiv_id)
        return paper.to_dict() if paper else None
    except ArxivFetchError as e:
        logger.error(f"Failed to fetch metadata for {arxiv_id}: {e}")
        raise


def ingest_to_corpus(
    arxiv_ids: List[str],
    corpus_dir: str = "corpus",
    storage_backend: str = "parquet",
    batch_size: int = 10,
) -> Dict[str, Any]:
    """
    Download and ingest arXiv papers into the corpus.

    This operation:
    1. Downloads PDFs for the specified arXiv papers
    2. Saves them to the corpus directory
    3. Runs the ingestion pipeline to chunk and index them
    4. Preserves append-only chunk ID assignment (existing IDs unchanged)

    Args:
        arxiv_ids: List of arXiv identifiers to ingest
        corpus_dir: Target corpus directory path (default: "corpus")
        storage_backend: Storage backend to use (parquet, memory, qdrant)
        batch_size: Number of papers to process per ingestion batch

    Returns:
        Dictionary containing:
        - attempted: Number of papers attempted
        - ingested: Number of papers successfully ingested
        - failed: List of failed arXiv IDs
        - corpus_dir: Corpus directory used
        - timestamp: ISO timestamp of operation
    """
    from src.ingestion.pipeline import IngestionPipeline

    fetcher = get_fetcher()
    corpus_path = Path(corpus_dir)
    corpus_path.mkdir(parents=True, exist_ok=True)

    # Initialize ingestion pipeline
    pipeline = IngestionPipeline(corpus_dir=corpus_path, storage_backend=storage_backend)

    # Ingest papers
    ingested_count, failed_ids = fetcher.ingest_to_corpus(
        arxiv_ids=arxiv_ids,
        corpus_dir=corpus_path,
        pipeline=pipeline,
        batch_size=batch_size,
    )

    return {
        "attempted": len(arxiv_ids),
        "ingested": ingested_count,
        "failed": failed_ids,
        "corpus_dir": str(corpus_path),
        "timestamp": datetime.now().isoformat(),
    }


# MCP Server Setup
if MCP_AVAILABLE:
    arxiv_server = Server("arxiv-corpus-manager")

    @arxiv_server.tool()
    async def mcp_search_arxiv(
        query: str,
        category: str = "cs.AI",
        max_results: int = 20,
        date_from: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """MCP tool wrapper for search_arxiv."""
        return search_arxiv(query, category, max_results, date_from)

    @arxiv_server.tool()
    async def mcp_fetch_arxiv_paper(arxiv_id: str) -> Optional[Dict[str, Any]]:
        """MCP tool wrapper for fetch_arxiv_paper."""
        return fetch_arxiv_paper(arxiv_id)

    @arxiv_server.tool()
    async def mcp_ingest_to_corpus(
        arxiv_ids: List[str],
        corpus_dir: str = "corpus",
        storage_backend: str = "parquet",
        batch_size: int = 10,
    ) -> Dict[str, Any]:
        """MCP tool wrapper for ingest_to_corpus."""
        return ingest_to_corpus(arxiv_ids, corpus_dir, storage_backend, batch_size)

    async def async_main():
        """Run the MCP server."""
        from mcp.server.stdio import stdio_server

        async with stdio_server() as (read_stream, write_stream):
            await arxiv_server.run(
                read_stream,
                write_stream,
                arxiv_server.create_initialization_options(),
            )


def main():
    """Entry point for running the MCP server."""
    if not MCP_AVAILABLE:
        print("Error: MCP package not installed. Install with: pip install mcp")
        return

    import asyncio

    asyncio.run(async_main())


if __name__ == "__main__":
    main()
