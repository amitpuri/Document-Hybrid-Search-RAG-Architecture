"""
src/evaluation/ablation.py

Graph-RAG factorial ablation harness.

WHY THIS EXISTS
----------------
Comparing this repo against Poulenard, Karmim & Barriere, "Knowledge-Graph
Based Augmentation versus Retrieval Augmented Generation for Cultural-
Related Question Answering" (arXiv:2609.18317) surfaced a gap: their
Table 3 varies exactly one factor at a time (encoder vs. none, textualized
graph vs. not, benchmark-aware extraction vs. not) so every accuracy delta
is attributable to a single component. This repo's own benchmark table
jumps straight from `rrf_dedup_mmr` (Strategy 8, no graph) to
`rrf_graph_dedup_mmr` (Strategy 14, full stack) -- so the MRR delta
between them is currently a bundle of three changes (graph fusion, dedup,
MMR), not a graph-attributable number on its own.

This harness adds three ablation cells to close that gap:

    graph_only         -- graph retrieval alone, no BM25/TF-IDF
    rrf_graph           -- RRF(BM25, TF-IDF, Graph), no post-processing
    rrf_graph_dedup     -- + Jaccard dedup, still no MMR

run alongside the six cells the repo already benchmarks, producing a full
one-factor-at-a-time ladder:

    bm25 -> tfidf -> rrf -> rrf_dedup -> rrf_dedup_mmr
                                       -> rrf_graph -> rrf_graph_dedup -> rrf_graph_dedup_mmr

INTEGRATION NOTE -- READ BEFORE RUNNING
-----------------------------------------
`graph_only`, `rrf_graph`, and `rrf_graph_dedup` are NOT yet registered in
`src/retrieval/pipeline.py`'s strategy dispatcher -- only
`rrf_graph_dedup_mmr` (Strategy 14) is currently wired up there. Until
that's patched, this harness still runs: it catches the failure for each
unregistered strategy and reports that cell as "not yet registered"
instead of crashing the whole run, so the other six cells are usable today.

Suggested patch for src/retrieval/pipeline.py (adjust names to match the
actual dispatcher -- this is written from the module names in README.md's
architecture diagram: retrievers/, fusion/, postprocessing/, not verified
against the file itself, since it wasn't available to generate this):

    # wherever the dispatcher currently has a branch resembling:
    #
    #     elif strategy == "rrf_graph_dedup_mmr":
    #         fused = reciprocal_rank_fusion([bm25_ranks, tfidf_ranks, graph_ranks])
    #         deduped = jaccard_dedup(fused)
    #         final = mmr_rerank(deduped, lambda_=DEFAULT_MMR_LAMBDA)
    #
    # add:
    #
    #     elif strategy == "graph_only":
    #         final = graph_ranks
    #     elif strategy == "rrf_graph":
    #         final = reciprocal_rank_fusion([bm25_ranks, tfidf_ranks, graph_ranks])
    #     elif strategy == "rrf_graph_dedup":
    #         final = jaccard_dedup(
    #             reciprocal_rank_fusion([bm25_ranks, tfidf_ranks, graph_ranks])
    #         )

USAGE
-----
    python -m src.evaluation.ablation
    python -m src.evaluation.ablation --corpus corpus --top-k 5 --out ablation_results.md

Built entirely on the public surface documented under "Python API Usage"
in README.md:

    engine = HybridSearchEngine.from_corpus("corpus/")
    engine.search(query, strategy=..., top_k=...)

See ROADMAP.md (item P0) for the full writeup this harness accompanies.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.engine import HybridSearchEngine
from src.evaluation.dataset import EVAL_DATASET

# Reuse the repo's own metric implementations when the names line up, so
# this harness stays consistent with whatever src/evaluation/metrics.py
# already does (e.g. its relevance-matching / RelCov logic). Falls back to
# small local implementations below if those names differ or the module
# isn't importable, so this script is never hard-blocked by that.
try:
    from src.evaluation import metrics as _repo_metrics
except ImportError:  # pragma: no cover
    _repo_metrics = None  # type: ignore[assignment]


# (strategy_alias, display_label, is_new_ablation_cell)
ABLATION_CELLS: List[Tuple[str, str, bool]] = [
    ("bm25", "Pure BM25 (Sparse)", False),
    ("tfidf", "Pure TF-IDF (Sparse Vector Space)", False),
    ("rrf", "RRF: BM25 + TF-IDF", False),
    ("rrf_dedup", "RRF + Dedup", False),
    ("rrf_dedup_mmr", "RRF + Dedup + MMR (Strategy 8)", False),
    ("graph_only", "Graph only, no BM25/TF-IDF", True),
    ("rrf_graph", "RRF: BM25 + TF-IDF + Graph, no post-proc", True),
    ("rrf_graph_dedup", "RRF + Dedup + Graph, no MMR", True),
    ("rrf_graph_dedup_mmr", "RRF + Dedup + MMR + Graph (Strategy 14)", False),
]


@dataclass
class CellResult:
    strategy: str
    label: str
    is_new_cell: bool
    available: bool = True
    error: Optional[str] = None
    per_query_rank: List[Optional[int]] = field(default_factory=list)
    mrr: float = 0.0
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    ndcg_at_5: float = 0.0
    entity_coverage: float = 0.0


def _find_target_rank(results: Sequence[Any], item: Dict[str, Any]) -> Optional[int]:
    """1-indexed rank of the ground-truth chunk in `results`, or None if absent.

    Primary match: chunk_id == target_chunk_idx (per the six-column chunk
    schema described in README.md: chunk_id, doc_name, page_num, section,
    text, metadata_json). Falls back to doc_name + entity substring match
    -- mirroring the substring fallback already noted in dataset.py's own
    module docstring -- for builds where chunk_id isn't exposed on the
    result object.
    """
    target_idx = item.get("target_chunk_idx")
    target_doc = item.get("target_doc", "")
    results = list(results)
    for i, res in enumerate(results):
        chunk = getattr(res, "chunk", res)
        rank = getattr(res, "rank", None) or (i + 1)
        chunk_id = getattr(chunk, "chunk_id", None)
        if chunk_id is not None and target_idx is not None and chunk_id == target_idx:
            return rank
        doc_name = getattr(chunk, "doc_name", "")
        text = getattr(chunk, "text", "") or ""
        if doc_name == target_doc and any(
            ent.lower() in text.lower() for ent in item.get("target_entities", [])
        ):
            return rank
    return None


def _reciprocal_rank(rank: Optional[int]) -> float:
    if _repo_metrics and hasattr(_repo_metrics, "reciprocal_rank"):
        return _repo_metrics.reciprocal_rank(rank)
    return 0.0 if rank is None else 1.0 / rank


def _recall_at_k(rank: Optional[int], k: int) -> float:
    if _repo_metrics and hasattr(_repo_metrics, "recall_at_k"):
        return _repo_metrics.recall_at_k(rank, k)
    return 1.0 if rank is not None and rank <= k else 0.0


def _ndcg_at_5(rank: Optional[int]) -> float:
    if _repo_metrics and hasattr(_repo_metrics, "ndcg_at_5"):
        return _repo_metrics.ndcg_at_5(rank)
    if rank is None or rank > 5:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def _entity_coverage(results: Sequence[Any], item: Dict[str, Any], top_k: int) -> float:
    """Fraction of ground-truth entities whose surface form appears
    anywhere in the top-k retrieved text -- a cheap proxy for the repo's
    RelCov metric (README's 'Key Empirical Findings' #4) when the
    canonical implementation isn't importable under this name."""
    if _repo_metrics and hasattr(_repo_metrics, "relation_coverage"):
        try:
            return _repo_metrics.relation_coverage(results, item)
        except Exception:
            pass
    entities = item.get("target_entities", [])
    if not entities:
        return 0.0
    blob = " ".join(
        (getattr(getattr(r, "chunk", r), "text", "") or "") for r in list(results)[:top_k]
    ).lower()
    hits = sum(1 for e in entities if e.lower() in blob)
    return hits / len(entities)


def run_ablation(
    corpus: str = "corpus",
    top_k: int = 5,
    dataset: Sequence[Dict[str, Any]] = EVAL_DATASET,
) -> List[CellResult]:
    """Run every cell in ABLATION_CELLS against `dataset` and return
    per-cell aggregate metrics. Cells whose strategy alias isn't
    registered in the pipeline dispatcher yet are marked unavailable
    rather than raising, so a single missing strategy doesn't abort the
    whole run."""
    engine = HybridSearchEngine.from_corpus(corpus)
    out: List[CellResult] = []

    for strategy, label, is_new in ABLATION_CELLS:
        cell = CellResult(strategy=strategy, label=label, is_new_cell=is_new)
        cached: List[Tuple[Dict[str, Any], Sequence[Any]]] = []

        for item in dataset:
            try:
                results = engine.search(item["query"], strategy=strategy, top_k=max(top_k, 5))
            except Exception as exc:  # noqa: BLE001 - broad on purpose: this is a
                # diagnostic harness, not production code. Any failure to run a
                # given strategy (e.g. unregistered alias) becomes a per-cell
                # "unavailable" result rather than ending the whole benchmark.
                cell.available = False
                cell.error = f"{type(exc).__name__}: {exc}"
                break
            cached.append((item, results))
            cell.per_query_rank.append(_find_target_rank(results, item))

        if cell.available and cached:
            n = len(cell.per_query_rank)
            cell.mrr = sum(_reciprocal_rank(r) for r in cell.per_query_rank) / n
            cell.recall_at_1 = sum(_recall_at_k(r, 1) for r in cell.per_query_rank) / n
            cell.recall_at_3 = sum(_recall_at_k(r, 3) for r in cell.per_query_rank) / n
            cell.recall_at_5 = sum(_recall_at_k(r, 5) for r in cell.per_query_rank) / n
            cell.ndcg_at_5 = sum(_ndcg_at_5(r) for r in cell.per_query_rank) / n
            cov = [_entity_coverage(results, item, top_k) for item, results in cached]
            cell.entity_coverage = sum(cov) / len(cov) if cov else 0.0

        out.append(cell)

    return out


def render_markdown(cells: List[CellResult]) -> str:
    lines = [
        "| Strategy | MRR | Recall@1 | Recall@3 | Recall@5 | NDCG@5 | EntCov | Status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in cells:
        if not c.available:
            status = "not yet registered" if c.is_new_cell else f"error: {c.error}"
            lines.append(f"| `{c.strategy}` | - | - | - | - | - | - | {status} |")
            continue
        tag = "[NEW] " if c.is_new_cell else ""
        lines.append(
            f"| {tag}`{c.strategy}` -- {c.label} | {c.mrr:.3f} | {c.recall_at_1:.3f} | "
            f"{c.recall_at_3:.3f} | {c.recall_at_5:.3f} | {c.ndcg_at_5:.3f} | "
            f"{c.entity_coverage:.3f} | ran |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Graph-RAG factorial ablation harness (see ROADMAP.md, item P0).",
    )
    parser.add_argument("--corpus", default="corpus", help="Corpus directory (default: corpus)")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k for Recall@k / NDCG@5 (default: 5)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write the markdown table to",
    )
    args = parser.parse_args()

    cells = run_ablation(corpus=args.corpus, top_k=args.top_k)
    table = render_markdown(cells)
    print(table)

    missing = [c for c in cells if c.is_new_cell and not c.available]
    if missing:
        names = ", ".join(f"`{c.strategy}`" for c in missing)
        print(
            f"\nNote: {names} are not yet registered in the strategy dispatcher. "
            "See ROADMAP.md (item P0) for the pipeline.py patch that enables them.",
            file=sys.stderr,
        )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(table + "\n")
        print(f"\nWrote table to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
