"""
Hybrid Document Search with Reciprocal Rank Fusion (RRF), Deduplication, and MMR.

Features:
1. Reciprocal Rank Fusion (RRF): Robust rank-based fusion (1 / (k + rank)) immune to outlier score skews.
2. Result-level Deduplication: Filters near-duplicate passages caused by chunk overlaps (Jaccard > 0.65).
3. Maximal Marginal Relevance (MMR): Balances top-relevance matches with diverse information coverage.
4. Uses structure-aware chunking and disk caching for instantaneous query responses.
"""

import os
import sys
import argparse
import numpy as np

# Reconfigure stdout for UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

from structure_aware_chunking import load_structured_corpus


# ===========================================================
# 1. Reciprocal Rank Fusion (RRF)
# ===========================================================
def reciprocal_rank_fusion(bm25_ranking, dense_ranking, k=60):
    """
    Combines sparse and dense rankings using Reciprocal Rank Fusion (RRF).
    Formula: RRF_score(d) = sum(1 / (k + rank(d) + 1))
    """
    scores = {}
    # BM25 rank contributions
    for rank, idx in enumerate(bm25_ranking):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

    # Dense rank contributions
    for rank, idx in enumerate(dense_ranking):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

    # Sort descending by RRF score
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    fused_indices = [idx for idx, _ in sorted_items]
    fused_scores = {idx: score for idx, score in sorted_items}
    return fused_indices, fused_scores


# ===========================================================
# 2. Result-Level Deduplication
# ===========================================================
def jaccard_similarity(tokens_a, tokens_b):
    """Computes Jaccard similarity between two sets of tokens."""
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def deduplicate_results(candidate_indices, corpus, threshold=0.65, max_results=10):
    """
    Filters candidate indices to remove redundant chunks resulting from sliding-window overlap.
    """
    selected_indices = []
    selected_token_sets = []

    for idx in candidate_indices:
        chunk_text = corpus[idx].lower()
        chunk_tokens = chunk_text.split()

        # Check overlap against all already accepted results
        is_duplicate = False
        for accepted_tokens in selected_token_sets:
            sim = jaccard_similarity(chunk_tokens, accepted_tokens)
            if sim >= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            selected_indices.append(idx)
            selected_token_sets.append(chunk_tokens)
            if len(selected_indices) >= max_results:
                break

    return selected_indices


# ===========================================================
# 3. Maximal Marginal Relevance (MMR) Re-Ranking
# ===========================================================
def maximal_marginal_relevance(
    query_vec,
    candidate_indices,
    corpus_vecs,
    relevance_scores,
    lambda_param=0.7,
    top_k=5
):
    r"""
    Applies MMR to balance relevance to query with diversity among selected documents.
    MMR = argmax_{d in R \ S} [ lambda * Sim(d, q) - (1 - lambda) * max_{s in S} Sim(d, s) ]
    """
    if not candidate_indices:
        return []

    # Map candidate indices to their pool
    pool = list(candidate_indices[:min(len(candidate_indices), top_k * 4)])
    selected = []

    # Pre-normalize relevance scores within the candidate pool
    pool_rel = np.array([relevance_scores.get(idx, 0.0) for idx in pool], dtype=np.float32)
    min_r, max_r = np.min(pool_rel), np.max(pool_rel)
    if max_r > min_r:
        pool_rel_norm = {idx: (r - min_r) / (max_r - min_r) for idx, r in zip(pool, pool_rel)}
    else:
        pool_rel_norm = {idx: 1.0 for idx in pool}

    while pool and len(selected) < top_k:
        if not selected:
            # Pick highest relevance candidate first
            best_idx = max(pool, key=lambda idx: pool_rel_norm[idx])
            selected.append(best_idx)
            pool.remove(best_idx)
            continue

        # Compute MMR score for each remaining candidate
        best_score = -float("inf")
        best_candidate = None

        # Dense vectors of already selected chunks
        selected_vecs = corpus_vecs[selected]

        for cand in pool:
            cand_vec = corpus_vecs[cand]
            rel_score = pool_rel_norm[cand]

            # Max similarity to already selected chunks
            sim_to_selected = float(np.max(cosine_similarity(cand_vec, selected_vecs)))

            # MMR formula
            mmr_score = (lambda_param * rel_score) - ((1.0 - lambda_param) * sim_to_selected)

            if mmr_score > best_score:
                best_score = mmr_score
                best_candidate = cand

        selected.append(best_candidate)
        pool.remove(best_candidate)

    return selected


# ===========================================================
# Main Search Pipeline
# ===========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Search with Reciprocal Rank Fusion (RRF), Deduplication, and MMR Diversity."
    )
    parser.add_argument(
        "query",
        nargs="*",
        default=["Binding", "Constraint", "Thesis", "in", "LLM", "agent", "execution", "harness"],
        help="Search query text",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=60,
        help="RRF smoothing constant k (default: 60)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to display (default: 5)",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        default=True,
        help="Enable result-level token overlap deduplication (default: True)",
    )
    parser.add_argument(
        "--mmr",
        action="store_true",
        help="Apply Maximal Marginal Relevance (MMR) diversity re-ranking",
    )
    parser.add_argument(
        "--lambda-param",
        type=float,
        default=0.7,
        help="MMR trade-off parameter between relevance and diversity (default: 0.7)",
    )
    args = parser.parse_args()

    active_query = " ".join(args.query)

    # 1. Load corpus (uses structure-aware chunking and disk cache)
    corpus_dir = os.path.join(os.path.dirname(__file__), "corpus")
    cache_dir = os.path.join(os.path.dirname(__file__), ".cache")
    corpus, total_pages, pdf_paths = load_structured_corpus(corpus_dir, cache_dir=cache_dir)

    print(f"Loaded {len(pdf_paths)} PDFs from: {corpus_dir}")
    print(f"Total Pages: {total_pages} | Chunks in Corpus: {len(corpus)}")
    print(f"Query: {active_query!r}")
    print(f"Mode: RRF (k={args.k}) | Dedup: {args.dedup} | MMR: {args.mmr}\n")

    # 2. Sparse Search: BM25Okapi
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    tokenized_query = active_query.lower().split()
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = np.array(bm25.get_scores(tokenized_query), dtype=np.float32)
    bm25_ranking = np.argsort(bm25_scores)[::-1]

    # 3. Dense Search: TF-IDF Cosine Similarity
    vectorizer = TfidfVectorizer(lowercase=True, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
    corpus_tfidf = vectorizer.fit_transform(corpus)
    query_tfidf = vectorizer.transform([active_query])
    dense_scores = cosine_similarity(query_tfidf, corpus_tfidf).flatten()
    dense_ranking = np.argsort(dense_scores)[::-1]

    # 4. Reciprocal Rank Fusion (RRF)
    rrf_ranking, rrf_scores = reciprocal_rank_fusion(bm25_ranking, dense_ranking, k=args.k)

    # 5. Deduplication & MMR Selection
    if args.dedup:
        # Pre-filter duplicates from candidate list
        filtered_candidates = deduplicate_results(rrf_ranking, corpus, threshold=0.65, max_results=args.top_k * 4)
    else:
        filtered_candidates = rrf_ranking

    if args.mmr:
        final_ranking = maximal_marginal_relevance(
            query_tfidf,
            filtered_candidates,
            corpus_tfidf,
            rrf_scores,
            lambda_param=args.lambda_param,
            top_k=args.top_k
        )
    else:
        final_ranking = filtered_candidates[:args.top_k]

    # 6. Display Comparison Table
    print(f"{'Rank':<5}{'RRF':<10}{'Cosine':<10}{'BM25':<10}Chunk Excerpt")
    print("-" * 110)
    for rank, idx in enumerate(final_ranking, start=1):
        raw_text = corpus[idx].replace("\n", " ")
        snippet = raw_text[:88] + "..." if len(raw_text) > 88 else raw_text
        print(
            f"{rank:<5}{rrf_scores.get(idx, 0.0):<10.4f}"
            f"{dense_scores[idx]:<10.3f}{bm25_scores[idx]:<10.3f}{snippet}"
        )

    # 7. Print Rank #1 Top Match in Full
    best_idx = final_ranking[0]
    print("\n" + "=" * 80)
    print(f"TOP MATCH (Rank #1) [RRF Score: {rrf_scores.get(best_idx, 0.0):.4f}]:")
    print("=" * 80)
    print(corpus[best_idx])


if __name__ == "__main__":
    main()
