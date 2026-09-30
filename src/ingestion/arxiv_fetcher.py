"""
arXiv Ingestion Pipeline - Rate-Limited arXiv API Client.

Implements Phase 4 of the roadmap: automated arXiv corpus ingestion with
category filtering, rate limiting, and append-only chunk ID stability.

Usage:
    from src.ingestion.arxiv_fetcher import ArxivFetcher, ArxivRateLimiter

    fetcher = ArxivFetcher(categories=["cs.AI"], rate_limiter=ArxivRateLimiter())
    papers = fetcher.search_papers("reinforcement learning", max_results=20)
    fetcher.ingest_to_corpus([p["id"] for p in papers], corpus_dir="corpus")
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)


class ArxivFetchError(Exception):
    """Base exception for arXiv fetch operations."""

    pass


class ArxivRateLimitError(ArxivFetchError):
    """Raised when arXiv API rate limit is exceeded."""

    pass


@dataclass
class ArxivPaper:
    """Represents an arXiv paper with metadata."""

    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    published: str
    categories: List[str]
    pdf_url: str
    primary_category: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "published": self.published,
            "categories": self.categories,
            "pdf_url": self.pdf_url,
            "primary_category": self.primary_category,
        }


class ArxivRateLimiter:
    """
    Rate limiter for arXiv API requests.

    Enforces minimum delay between requests (default 3 seconds per arXiv API terms)
    with exponential backoff on HTTP 429 errors.
    """

    def __init__(
        self,
        min_delay: float = 3.0,
        max_retries: int = 5,
        base_backoff: float = 2.0,
        max_backoff: float = 60.0,
    ):
        self.min_delay = min_delay
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self.last_request_time: float = 0.0
        self.consecutive_failures: int = 0

    def wait(self) -> None:
        """Wait the required delay before making the next request."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_delay:
            sleep_time = self.min_delay - elapsed
            logger.debug(f"Rate limiter: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def record_success(self) -> None:
        """Record a successful request, resetting failure counter."""
        self.consecutive_failures = 0

    def record_failure(self) -> float:
        """
        Record a failed request and return the backoff delay.

        Implements exponential backoff: delay = base_backoff * (2^failures)
        capped at max_backoff.
        """
        self.consecutive_failures += 1
        if self.consecutive_failures > self.max_retries:
            raise ArxivRateLimitError(f"Max retries ({self.max_retries}) exceeded for arXiv API")
        backoff = min(
            self.base_backoff * (2 ** (self.consecutive_failures - 1)),
            self.max_backoff,
        )
        logger.warning(
            f"Rate limiter: backing off {backoff:.2f}s "
            f"(failure {self.consecutive_failures}/{self.max_retries})"
        )
        return backoff

    def reset(self) -> None:
        """Reset the rate limiter state."""
        self.last_request_time = 0.0
        self.consecutive_failures = 0


class ArxivFetcher:
    """
    Rate-limited arXiv API client for paper discovery and PDF ingestion.

    Supports:
    - Category-filtered search (e.g., cs.AI, cs.IR, cs.CL)
    - Rate limiting with exponential backoff
    - PDF download and ingestion into existing corpus
    - Append-only chunk ID assignment via IngestionPipeline
    """

    def __init__(
        self,
        categories: Optional[List[str]] = None,
        rate_limiter: Optional[ArxivRateLimiter] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.categories = categories or ["cs.AI"]
        self.rate_limiter = rate_limiter or ArxivRateLimiter()
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".cache/arxiv")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Lazy import arxiv to avoid hard dependency
        try:
            import arxiv

            self.arxiv_client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=5)
        except ImportError:
            raise ImportError(
                "arxiv package is required for arXiv ingestion. " "Install with: pip install arxiv"
            )

    def search_papers(
        self,
        query: str,
        max_results: int = 100,
        date_from: Optional[str] = None,
        sort_by: str = "relevance",
    ) -> List[ArxivPaper]:
        """
        Search arXiv for papers matching query and category filters.

        Args:
            query: Search query string (e.g., "reinforcement learning")
            max_results: Maximum number of results to return
            date_from: Optional date filter (YYYY-MM-DD format)
            sort_by: Sort order: "relevance", "lastUpdatedDate", "submittedDate"

        Returns:
            List of ArxivPaper objects matching the criteria
        """
        import arxiv

        # Build category filter
        cat_query = " OR ".join([f"cat:{cat}" for cat in self.categories])
        full_query = f"({cat_query}) AND ({query})" if query else cat_query

        logger.info(f"Searching arXiv: {full_query} (max_results={max_results})")

        # Map sort_by string to arxiv.SortCriterion
        sort_map = {
            "relevance": arxiv.SortCriterion.Relevance,
            "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
            "submittedDate": arxiv.SortCriterion.SubmittedDate,
        }
        sort_criterion = sort_map.get(sort_by, arxiv.SortCriterion.Relevance)

        # Build search
        search = arxiv.Search(
            query=full_query,
            max_results=max_results,
            sort_by=sort_criterion,
            sort_order=arxiv.SortOrder.Descending,
        )

        # Apply date filter if specified
        if date_from:
            # Note: arxiv library doesn't support date filtering directly in Search
            # We'll filter results after fetching
            try:
                date_obj = datetime.strptime(date_from, "%Y-%m-%d")
            except ValueError:
                logger.warning(f"Invalid date format: {date_from}, ignoring date filter")
                date_obj = None

        papers: List[ArxivPaper] = []
        self.rate_limiter.wait()

        try:
            for result in self.arxiv_client.results(search):
                # Date filter if specified
                if date_from and date_obj:
                    if result.published.replace(tzinfo=timezone.utc) < date_obj:
                        continue

                paper = ArxivPaper(
                    arxiv_id=result.entry_id.split("/")[-1],
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    abstract=result.summary,
                    published=(result.published.isoformat() if result.published else ""),
                    categories=result.categories,
                    pdf_url=result.pdf_url,
                    primary_category=result.primary_category,
                )
                papers.append(paper)
                self.rate_limiter.record_success()

        except Exception as e:
            backoff = self.rate_limiter.record_failure()
            time.sleep(backoff)
            raise ArxivFetchError(f"arXiv search failed: {e}") from e

        logger.info(f"Found {len(papers)} papers matching criteria")
        return papers

    def fetch_paper_pdf(self, arxiv_id: str) -> Optional[bytes]:
        """
        Download PDF for a specific arXiv paper.

        Args:
            arxiv_id: arXiv identifier (e.g., "2301.07041")

        Returns:
            PDF content as bytes, or None if download fails
        """
        import arxiv

        # Check cache first
        cache_path = self.cache_dir / f"{arxiv_id}.pdf"
        if cache_path.exists():
            logger.debug(f"Using cached PDF for {arxiv_id}")
            return cache_path.read_bytes()

        # Fetch from arXiv
        self.rate_limiter.wait()
        try:
            paper = next(self.arxiv_client.results(arxiv.Search(id_list=[arxiv_id])))
            pdf_url = paper.pdf_url

            # Download PDF using urllib (arxiv v4.0+ API)
            pdf_path = self.cache_dir / f"{arxiv_id}.pdf"
            urlretrieve(pdf_url, str(pdf_path))

            if pdf_path.exists():
                content = pdf_path.read_bytes()
                logger.info(f"Downloaded PDF for {arxiv_id} ({len(content)} bytes)")
                return content
            else:
                logger.error(f"Failed to download PDF for {arxiv_id}")
                return None

        except Exception as e:
            backoff = self.rate_limiter.record_failure()
            time.sleep(backoff)
            logger.error(f"Failed to fetch PDF for {arxiv_id}: {e}")
            return None

    def ingest_to_corpus(
        self,
        arxiv_ids: List[str],
        corpus_dir: Path,
        pipeline: Any,
        batch_size: int = 10,
    ) -> Tuple[int, List[str]]:
        """
        Download and ingest arXiv papers into the corpus.

        Args:
            arxiv_ids: List of arXiv identifiers to ingest
            corpus_dir: Target corpus directory
            pipeline: IngestionPipeline instance for chunking
            batch_size: Number of papers to download before batch ingestion

        Returns:
            Tuple of (papers_ingested, failed_arxiv_ids)
        """
        corpus_dir = Path(corpus_dir)
        corpus_dir.mkdir(parents=True, exist_ok=True)

        ingested_count = 0
        failed_ids: List[str] = []

        for i, arxiv_id in enumerate(arxiv_ids):
            logger.info(f"Processing {arxiv_id} ({i+1}/{len(arxiv_ids)})")

            # Download PDF
            pdf_content = self.fetch_paper_pdf(arxiv_id)
            if pdf_content is None:
                failed_ids.append(arxiv_id)
                continue

            # Save PDF to corpus
            pdf_filename = f"{arxiv_id}.pdf"
            pdf_path = corpus_dir / pdf_filename
            pdf_path.write_bytes(pdf_content)

            logger.info(f"Saved {pdf_filename} to corpus")

            # Batch ingestion: process every batch_size papers
            if (i + 1) % batch_size == 0 or i == len(arxiv_ids) - 1:
                try:
                    logger.info("Running ingestion pipeline on corpus...")
                    chunk_store, total_pages, pdf_paths = pipeline.run(force_rebuild=False)
                    ingested_count = len(pdf_paths)
                    logger.info(f"Ingestion complete: {ingested_count} PDFs, {total_pages} pages")
                except Exception as e:
                    logger.error(f"Ingestion failed: {e}")
                    # Continue with next batch even if this one fails

        return ingested_count, failed_ids

    def get_paper_metadata(self, arxiv_id: str) -> Optional[ArxivPaper]:
        """
        Get metadata for a specific arXiv paper without downloading PDF.

        Args:
            arxiv_id: arXiv identifier

        Returns:
            ArxivPaper object or None if not found
        """
        import arxiv

        self.rate_limiter.wait()
        try:
            paper = next(self.arxiv_client.results(arxiv.Search(id_list=[arxiv_id])))
            arxiv_paper = ArxivPaper(
                arxiv_id=paper.entry_id.split("/")[-1],
                title=paper.title,
                authors=[author.name for author in paper.authors],
                abstract=paper.summary,
                published=(paper.published.isoformat() if paper.published else ""),
                categories=paper.categories,
                pdf_url=paper.pdf_url,
                primary_category=paper.primary_category,
            )
            self.rate_limiter.record_success()
            return arxiv_paper

        except Exception as e:
            backoff = self.rate_limiter.record_failure()
            time.sleep(backoff)
            logger.error(f"Failed to fetch metadata for {arxiv_id}: {e}")
            return None
