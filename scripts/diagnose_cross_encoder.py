"""
Cross-Encoder Diagnostic Trace
===============================
Sanity-checks the cross-encoder retriever against a single query by printing:
  1. RRF top-20 pool entering the reranker (index, text preview, original RRF rank)
  2. Raw cross-encoder scores for each candidate
  3. Before vs. after ranking side-by-side
  4. BM25 rank vs. cross-encoder rank for the known ground-truth chunk

Usage:
    python scripts/diagnose_cross_encoder.py
    python scripts/diagnose_cross_encoder.py --query "POMDP belief state filtering" --gt-doc "pomdp"
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.console import ensure_utf8_streams

# Ensure UTF-8 output on Windows
ensure_utf8_streams()

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.engine import HybridSearchEngine
from src.config import CORPUS_DIR, DEFAULT_CROSS_ENCODER_POOL_SIZE, DEFAULT_RRF_K
from src.retrieval.fusion.rrf import reciprocal_rank_fusion


# ── Configurable defaults ────────────────────────────────────────────────────
DEFAULT_QUERY = "StarShell operator configuration"
DEFAULT_GT_DOC = "starshell"   # substring of the expected ground-truth doc_name
POOL_SIZE = DEFAULT_CROSS_ENCODER_POOL_SIZE  # 50


def find_gt_rank(ranked_indices, chunk_store, gt_doc_substr: str) -> int | None:
    """Returns the 1-based rank of the first chunk whose doc_name contains gt_doc_substr."""
    for rank, idx in enumerate(ranked_indices, start=1):
        chunk = chunk_store.get_chunk(idx)
        if chunk and gt_doc_substr.lower() in chunk.doc_name.lower():
            return rank
    return None


def main():
    parser = argparse.ArgumentParser(description="Cross-Encoder Diagnostic Trace")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Query to diagnose")
    parser.add_argument(
        "--gt-doc", default=DEFAULT_GT_DOC,
        help="Substring of expected ground-truth doc_name (for rank comparison)"
    )
    parser.add_argument("--corpus", default=str(CORPUS_DIR), help="Corpus directory")
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f" Cross-Encoder Diagnostic Trace")
    print(f" Query : {args.query!r}")
    print(f" GT doc: contains {args.gt_doc!r}")
    print(f"{'='*80}\n")

    # ── Load engine ──────────────────────────────────────────────────────────
    print("Loading corpus and indexing…")
    engine = HybridSearchEngine.from_corpus(corpus_dir=args.corpus)
    rp = engine.retrieval
    chunk_store = rp.chunk_store
    corpus_texts = rp.corpus_texts

    # ── Base scores ──────────────────────────────────────────────────────────
    b_scores = rp.bm25.score(args.query)
    d_scores = rp.tfidf.score(args.query)
    b_rank = np.argsort(b_scores)[::-1].tolist()
    d_rank = np.argsort(d_scores)[::-1].tolist()

    rrf_wide, _ = reciprocal_rank_fusion(b_rank, d_rank, k=DEFAULT_RRF_K)

    # ── RRF pool entering cross-encoder ──────────────────────────────────────
    pool = rrf_wide[:POOL_SIZE]
    remainder = rrf_wide[POOL_SIZE:]

    print(f"RRF top-{POOL_SIZE} pool (candidates entering cross-encoder):\n")
    print(f"  {'RRF Rank':<10}{'BM25 Rank':<11}{'Chunk Idx':<11}{'Doc / Page / Section'}")
    print(f"  {'-'*8:<10}{'-'*9:<11}{'-'*9:<11}{'-'*45}")
    for rrf_rank, idx in enumerate(pool, start=1):
        chunk = chunk_store.get_chunk(idx)
        bm25_rank = b_rank.index(idx) + 1 if idx in b_rank[:200] else ">200"
        if chunk:
            label = f"{chunk.doc_name} | p{chunk.page_num} | §{chunk.section}"
        else:
            label = "(unknown)"
        print(f"  {rrf_rank:<10}{str(bm25_rank):<11}{idx:<11}{label}")

    # ── Raw cross-encoder scores ─────────────────────────────────────────────
    if rp.cross_encoder is None or rp.cross_encoder.model is None:
        print("\n[ERROR] Cross-encoder model not loaded. Cannot continue diagnosis.")
        sys.exit(1)

    print(f"\nCross-encoder raw scores for pool of {len(pool)} candidates:\n")
    pairs = [(args.query, corpus_texts[idx]) for idx in pool]
    ce_scores = rp.cross_encoder.model.predict(pairs)

    sorted_by_ce = sorted(zip(ce_scores, pool), reverse=True)

    print(f"  {'CE Rank':<9}{'CE Score':<12}{'RRF Rank':<10}{'Chunk Idx':<11}{'Doc / Page'}")
    print(f"  {'-'*7:<9}{'-'*10:<12}{'-'*8:<10}{'-'*9:<11}{'-'*35}")
    for ce_rank, (score, idx) in enumerate(sorted_by_ce, start=1):
        rrf_rank = pool.index(idx) + 1
        chunk = chunk_store.get_chunk(idx)
        label = f"{chunk.doc_name} | p{chunk.page_num}" if chunk else "(unknown)"
        print(f"  {ce_rank:<9}{score:<12.4f}{rrf_rank:<10}{idx:<11}{label}")

    # ── Before vs after comparison ───────────────────────────────────────────
    ce_reranked = [idx for _, idx in sorted_by_ce]

    print(f"\nBefore vs After ranking (top 10):\n")
    print(f"  {'#':<5}  {'RRF (before)':<35}  {'Cross-Encoder (after)'}")
    print(f"  {'-'*3:<5}  {'-'*33:<35}  {'-'*33}")
    for i in range(min(10, len(pool))):
        rrf_idx = pool[i]
        ce_idx = ce_reranked[i]
        rrf_chunk = chunk_store.get_chunk(rrf_idx)
        ce_chunk = chunk_store.get_chunk(ce_idx)
        rrf_label = f"[{rrf_idx}] {rrf_chunk.doc_name[:28]}" if rrf_chunk else f"[{rrf_idx}]"
        ce_label = f"[{ce_idx}] {ce_chunk.doc_name[:28]}" if ce_chunk else f"[{ce_idx}]"
        print(f"  {i+1:<5}  {rrf_label:<35}  {ce_label}")

    # ── Ground-truth rank comparison ─────────────────────────────────────────
    bm25_gt_rank = find_gt_rank(b_rank, chunk_store, args.gt_doc)
    rrf_gt_rank  = find_gt_rank(rrf_wide, chunk_store, args.gt_doc)
    ce_gt_rank   = find_gt_rank(ce_reranked + remainder, chunk_store, args.gt_doc)

    print(f"\nGround-Truth Chunk Rank Summary (doc contains {args.gt_doc!r}):\n")
    print(f"  BM25 rank       : {bm25_gt_rank or 'not in top-2000'}")
    print(f"  RRF rank        : {rrf_gt_rank or 'not in top-2000'}")
    print(f"  Cross-Enc rank  : {ce_gt_rank or 'not in ranking'}")

    # ── Automated conclusion ─────────────────────────────────────────────────
    print(f"\n{'='*80}")
    score_range = ce_scores.max() - ce_scores.min()
    top_ce_score = ce_scores.max()
    bottom_ce_score = ce_scores.min()

    if score_range < 0.05:
        conclusion = "FLAT SCORES — cross-encoder outputting near-uniform scores (possible model load issue)"
    elif rrf_gt_rank and ce_gt_rank and ce_gt_rank > rrf_gt_rank * 2:
        if top_ce_score < 0:
            conclusion = "DOMAIN MISMATCH LIKELY — scores mostly negative (ms-marco calibrated for web text, not scientific jargon)"
        else:
            conclusion = "DOMAIN MISMATCH CONFIRMED — GT rank regressed vs RRF; scores not obviously inverted"
    elif rrf_gt_rank and ce_gt_rank and ce_gt_rank < rrf_gt_rank:
        conclusion = "CROSS-ENCODER HELPING — GT rank improved vs RRF"
    else:
        conclusion = "INCONCLUSIVE — inspect scores manually above"

    print(f" Score range: {bottom_ce_score:.4f} → {top_ce_score:.4f}")
    print(f" Conclusion : {conclusion}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
