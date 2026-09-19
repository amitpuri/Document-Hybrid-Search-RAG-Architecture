"""
Information Retrieval Evaluation Harness for Document Hybrid Search.

Computes standard IR metrics across a curated benchmark query set with ground-truth targets:
- MRR (Mean Reciprocal Rank)
- Recall@1, Recall@3, Recall@5
- NDCG@5 (Normalized Discounted Cumulative Gain)

Evaluates and compares:
1.  Pure BM25 (Sparse Keyword Retrieval)
2.  Pure Dense TF-IDF (Vector Space Model)
3.  Linear Hybrid Fusion (Alpha = 0.3)
4.  Linear Hybrid Fusion (Alpha = 0.5)
5.  Linear Hybrid Fusion (Alpha = 0.7)
6.  Reciprocal Rank Fusion (RRF k=60)
7.  RRF + Deduplication
8.  RRF + Dedup + MMR
9.  PPMI Distributional Semantic + BM25 RRF
"""

import os
import sys
import math
import numpy as np

# Reconfigure stdout for UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

from corpus_loader import load_structured_corpus
from hybrid_search_rrf import reciprocal_rank_fusion, maximal_marginal_relevance, deduplicate_results
from pmi_semantic_search_from_scratch import (
    PPMIEmbeddings,
    BM25 as BM25Scratch,
    rrf_fusion,
    sparse_cosine_similarity,
    tokenize as ppmi_tokenize,
)


# ===========================================================
# Curated Evaluation Set (Queries with Ground-Truth Targets)
# ===========================================================
EVAL_DATASET = [
    {
        "query": "Binding Constraint Thesis in LLM agent execution harness",
        "target_doc": "2605.23950v1.pdf",
    },
    {
        "query": "Terminal Agents Suffice for Enterprise Automation StarShell command line",
        "target_doc": "2604.00073v3.pdf",
    },
    {
        "query": "Dynamic Tiered AgentRunner Framework Risk Adaptive Tiering",
        "target_doc": "2605.10223v1.pdf",
    },
    {
        "query": "Position AI Evaluations Should be Grounded on a Theory of Capability",
        "target_doc": "2509.19590v2.pdf",
    },
    {
        "query": "Economy of AI agents market forces and firm sizes",
        "target_doc": "2509.01063v1.pdf",
    },
    {
        "query": "Partially Observed Markov Decision Processes belief state filtering",
        "target_doc": "1604.08127v1.pdf",
    },
    {
        "query": "Thinking with Looped Flows recurrent reasoning depth",
        "target_doc": "2609.11801v1.pdf",
    },
    {
        "query": "Can coding agents be general agents web browser interaction",
        "target_doc": "2604.13107v1.pdf",
    },
    {
        "query": "Code as Agent Harness context window constraint",
        "target_doc": "2605.18747v1.pdf",
    },
    {
        "query": "Your model already knows hidden representation extraction",
        "target_doc": "2609.11310v1.pdf",
    },
]


# ===========================================================
# Relevance Signal
# Chunks are prefixed with their source filename by the chunker:
#   "[2605.23950v1.pdf | Page 3 | § Introduction] ..."
# Checking the first 120 chars is robust against paraphrase and
# does not require brittle key-phrase substring matching.
# ===========================================================
def is_relevant(chunk_text, target_doc):
    """Returns True if the chunk originated from target_doc."""
    return target_doc.lower() in chunk_text[:120].lower()


def evaluate_ranking(ranked_indices, corpus, target_doc, k_list=(1, 3, 5)):
    """Computes MRR, Recall@K, and NDCG@5 for a single query."""
    relevance_flags = [is_relevant(corpus[idx], target_doc) for idx in ranked_indices]

    # MRR
    reciprocal_rank = 0.0
    for rank, rel in enumerate(relevance_flags, start=1):
        if rel:
            reciprocal_rank = 1.0 / rank
            break

    # Recall@K
    recalls = {k: 1.0 if any(relevance_flags[:k]) else 0.0 for k in k_list}

    # NDCG@5
    dcg = sum(
        (1.0 if rel else 0.0) / math.log2(rank + 1)
        for rank, rel in enumerate(relevance_flags[:5], start=1)
    )
    idcg = 1.0 / math.log2(2)  # Ideal: 1 relevant doc at rank 1
    ndcg_5 = dcg / idcg if idcg > 0 else 0.0

    return {
        "mrr": reciprocal_rank,
        "recall_1": recalls.get(1, 0.0),
        "recall_3": recalls.get(3, 0.0),
        "recall_5": recalls.get(5, 0.0),
        "ndcg_5": ndcg_5,
    }


# ===========================================================
# Strategy Runner Helpers (pure functions, no side-effects)
# ===========================================================
def _linear_fusion(b_scores, d_scores, alpha):
    b_norm = (b_scores - b_scores.min()) / (b_scores.max() - b_scores.min() + 1e-9)
    d_norm = (d_scores - d_scores.min()) / (d_scores.max() - d_scores.min() + 1e-9)
    hybrid = (alpha * d_norm) + ((1.0 - alpha) * b_norm)
    return np.argsort(hybrid)[::-1].tolist()


def _rrf(b_scores, d_scores, k=60):
    b_rank = np.argsort(b_scores)[::-1]
    d_rank = np.argsort(d_scores)[::-1]
    fused_idx, _ = reciprocal_rank_fusion(b_rank, d_rank, k=k)
    return fused_idx


def _rrf_dedup(b_scores, d_scores, corpus, k=60):
    fused_idx = _rrf(b_scores, d_scores, k=k)
    return deduplicate_results(fused_idx, corpus, threshold=0.65, max_results=10)


def _rrf_dedup_mmr(q_vec, b_scores, d_scores, corpus_tfidf, corpus, k=60):
    b_rank = np.argsort(b_scores)[::-1]
    d_rank = np.argsort(d_scores)[::-1]
    fused_idx, fused_scores = reciprocal_rank_fusion(b_rank, d_rank, k=k)
    deduped = deduplicate_results(fused_idx, corpus, threshold=0.65, max_results=20)
    return maximal_marginal_relevance(
        q_vec, deduped, corpus_tfidf, fused_scores, lambda_param=0.7, top_k=5
    )


# ===========================================================
# Benchmark Harness
# ===========================================================
def run_evaluation():
    base_dir = os.path.dirname(__file__)
    corpus_dir = os.path.join(base_dir, "corpus")
    cache_dir = os.path.join(base_dir, ".cache")

    print(f"Loading corpus from: {corpus_dir}")
    corpus, total_pages, pdf_paths = load_structured_corpus(corpus_dir, cache_dir=cache_dir)
    print(f"Loaded {len(pdf_paths)} PDFs | Total Pages: {total_pages} | Chunks: {len(corpus)}\n")

    # ----------------------------------------------------------
    # Pre-compute shared models (built once, reused per query)
    # ----------------------------------------------------------
    print("Precomputing BM25 and TF-IDF models...")
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    vectorizer = TfidfVectorizer(
        lowercase=True, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b"
    )
    corpus_tfidf = vectorizer.fit_transform(corpus)

    # PPMI model (built once; ~20-40 s on a 1000-chunk corpus)
    print("Building PPMI distributional embeddings (window=5, vocab=1500)...")
    ppmi_tokens = [ppmi_tokenize(doc) for doc in corpus]
    embedder = PPMIEmbeddings(ppmi_tokens, window_size=5, vocab_size=1500, max_context_per_word=50)
    bm25_scratch = BM25Scratch(ppmi_tokens)
    chunk_embeddings = [embedder.embed_text(toks) for toks in ppmi_tokens]
    print(f"PPMI vocab: {len(embedder.vocab)} terms | Embeddings: {len(chunk_embeddings)}\n")

    # ----------------------------------------------------------
    # Metric accumulators
    # ----------------------------------------------------------
    strategy_names = [
        "1. Pure BM25 (Sparse)",
        "2. Pure TF-IDF (Dense)",
        "3. Linear Hybrid (a=0.3)",
        "4. Linear Hybrid (a=0.5)",
        "5. Linear Hybrid (a=0.7)",
        "6. RRF (k=60)",
        "7. RRF + Deduplication",
        "8. RRF + Dedup + MMR",
        "9. PPMI Semantic + BM25 RRF",
    ]
    results = {name: [] for name in strategy_names}

    print(f"Evaluating {len(EVAL_DATASET)} queries x {len(strategy_names)} strategies...\n")

    for item in EVAL_DATASET:
        q_text = item["query"]
        target_doc = item["target_doc"]

        # --- Shared retrieval signals ---
        q_tokens = q_text.lower().split()
        b_scores = np.array(bm25.get_scores(q_tokens), dtype=np.float32)

        q_vec = vectorizer.transform([q_text])
        d_scores = cosine_similarity(q_vec, corpus_tfidf).flatten().astype(np.float32)

        # PPMI signal
        q_ppmi_toks = ppmi_tokenize(q_text)
        q_ppmi_vec, q_ppmi_mag = embedder.embed_text(q_ppmi_toks)
        ppmi_scores = [
            sparse_cosine_similarity(q_ppmi_vec, q_ppmi_mag, c_vec, c_mag)
            for c_vec, c_mag in chunk_embeddings
        ]
        ppmi_bm25_scores = bm25_scratch.get_scores(q_ppmi_toks)
        ppmi_ranking = sorted(range(len(corpus)), key=lambda i: ppmi_scores[i], reverse=True)
        ppmi_bm25_ranking = sorted(
            range(len(corpus)), key=lambda i: ppmi_bm25_scores[i], reverse=True
        )
        ppmi_fused, _ = rrf_fusion(ppmi_ranking, ppmi_bm25_ranking, k=60)

        # --- Single dispatch path: build ranked list for every strategy ---
        ranked_by_strategy = {
            "1. Pure BM25 (Sparse)":       np.argsort(b_scores)[::-1].tolist(),
            "2. Pure TF-IDF (Dense)":       np.argsort(d_scores)[::-1].tolist(),
            "3. Linear Hybrid (a=0.3)":     _linear_fusion(b_scores, d_scores, 0.3),
            "4. Linear Hybrid (a=0.5)":     _linear_fusion(b_scores, d_scores, 0.5),
            "5. Linear Hybrid (a=0.7)":     _linear_fusion(b_scores, d_scores, 0.7),
            "6. RRF (k=60)":               _rrf(b_scores, d_scores, k=60),
            "7. RRF + Deduplication":       _rrf_dedup(b_scores, d_scores, corpus, k=60),
            "8. RRF + Dedup + MMR":         _rrf_dedup_mmr(
                                                q_vec, b_scores, d_scores, corpus_tfidf, corpus
                                            ),
            "9. PPMI Semantic + BM25 RRF": ppmi_fused,
        }

        for name in strategy_names:
            ranked = ranked_by_strategy[name]
            metrics = evaluate_ranking(ranked, corpus, target_doc)
            results[name].append(metrics)

    # ----------------------------------------------------------
    # Print Summary Table
    # ----------------------------------------------------------
    print(
        f"\n{'Retrieval Strategy':<32}"
        f"{'MRR':<10}{'Recall@1':<12}{'Recall@3':<12}{'Recall@5':<12}{'NDCG@5':<10}"
    )
    print("=" * 90)

    for name in strategy_names:
        ml = results[name]
        print(
            f"{name:<32}"
            f"{np.mean([m['mrr']      for m in ml]):<10.3f}"
            f"{np.mean([m['recall_1'] for m in ml]):<12.3f}"
            f"{np.mean([m['recall_3'] for m in ml]):<12.3f}"
            f"{np.mean([m['recall_5'] for m in ml]):<12.3f}"
            f"{np.mean([m['ndcg_5']   for m in ml]):<10.3f}"
        )

    print()


if __name__ == "__main__":
    run_evaluation()

