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
10. Cross-Encoder Re-rank (wide pre-dedup pool of 50)
11. Sentence-Transformer Dense Retrieval (MiniLM)
12. Adaptive Hybrid (Query-Type Alpha Heuristic)
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

from corpus_loader import load_structured_corpus, compute_cache_key
from hybrid_search_rrf import reciprocal_rank_fusion, maximal_marginal_relevance, deduplicate_results
from pmi_semantic_search_from_scratch import (
    PPMIEmbeddings,
    BM25 as BM25Scratch,
    rrf_fusion,
    sparse_cosine_similarity,
    tokenize as ppmi_tokenize,
)

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    HAS_NEURAL = True
except ImportError:
    HAS_NEURAL = False


# ===========================================================
# Curated Evaluation Set (Queries with Ground-Truth Targets)
# ===========================================================
EVAL_DATASET = [
    {
        "query": "Binding Constraint Thesis in LLM agent execution harness",
        "target_doc": "2605.23950v1.pdf",
        "target_chunk_idx": 1561,  # § 3 The Binding Constraint Thesis
    },
    {
        "query": "Terminal Agents Suffice for Enterprise Automation StarShell command line",
        "target_doc": "2604.00073v3.pdf",
        "target_chunk_idx": 564,   # § 3.2 StarShell: A Terminal-Based Enterprise Agent
    },
    {
        "query": "Dynamic Tiered AgentRunner Framework Risk Adaptive Tiering",
        "target_doc": "2605.10223v1.pdf",
        "target_chunk_idx": 865,   # § Abstract / Framework intro
    },
    {
        "query": "Position AI Evaluations Should be Grounded on a Theory of Capability",
        "target_doc": "2509.19590v2.pdf",
        "target_chunk_idx": 396,   # § Abstract / Position definition
    },
    {
        "query": "Economy of AI agents market forces and firm sizes",
        "target_doc": "2509.01063v1.pdf",
        "target_chunk_idx": 321,   # § 3 Organizations of AI agents / 3.1 Firm sizes
    },
    {
        "query": "Partially Observed Markov Decision Processes belief state filtering",
        "target_doc": "1604.08127v1.pdf",
        "target_chunk_idx": 0,     # § Overview / POMDP intro
    },
    {
        "query": "Thinking with Looped Flows recurrent reasoning depth",
        "target_doc": "2609.11801v1.pdf",
        "target_chunk_idx": 1963,  # § 1 Contributions / Looped flows framework
    },
    {
        "query": "Can coding agents be general agents web browser interaction",
        "target_doc": "2604.13107v1.pdf",
        "target_chunk_idx": 820,   # § 1 Title & Abstract
    },
    {
        "query": "Code as Agent Harness context window constraint",
        "target_doc": "2605.18747v1.pdf",
        "target_chunk_idx": 1149,  # § 4.1 Context Window Constraint
    },
    {
        "query": "Your model already knows hidden representation extraction",
        "target_doc": "2609.11310v1.pdf",
        "target_chunk_idx": 1725,  # § 1 Title & Abstract
    },
    # --- NL-phrased queries: designed to stress-test Strategy 12's α=0.7 branch ---
    # Each contains an NL-indicator word (how/what/can) and >4 tokens.
    # Reuse validated target_chunk_idx values — no re-labelling needed.
    {
        "query": "How does the Binding Constraint Thesis affect harness comparisons across models",
        "target_doc": "2605.23950v1.pdf",
        "target_chunk_idx": 1561,  # same target as query 1, NL phrasing → α=0.7
    },
    {
        "query": "What risk-tiering mechanisms does the AgentRunner framework apply",
        "target_doc": "2605.10223v1.pdf",
        "target_chunk_idx": 865,   # same target as query 3, NL phrasing → α=0.7
    },
    {
        "query": "How do POMDP belief states update after receiving new observations",
        "target_doc": "1604.08127v1.pdf",
        "target_chunk_idx": 0,     # same target as query 6, NL phrasing → α=0.7
    },
    {
        "query": "What market forces shape the organization and size of AI agent firms",
        "target_doc": "2509.01063v1.pdf",
        "target_chunk_idx": 321,   # same target as query 5, NL phrasing → α=0.7
    },
]


# ===========================================================
# Relevance Signal
# Fine-grained chunk-level relevance evaluates whether retrieval
# located the specific informative passage rather than a lucky
# hit on references, captions, or author lists.
# Falls back to doc prefix match if target_chunk_idx is None.
# ===========================================================
def is_relevant(chunk_text, chunk_idx, target_doc, target_chunk_idx=None):
    """Returns True if the chunk matches target_chunk_idx (or doc prefix fallback)."""
    if target_chunk_idx is not None:
        if isinstance(target_chunk_idx, (set, list, tuple)):
            return chunk_idx in target_chunk_idx
        return chunk_idx == target_chunk_idx
    return target_doc.lower() in chunk_text[:120].lower()


def evaluate_ranking(ranked_indices, corpus, target_doc, target_chunk_idx=None, k_list=(1, 3, 5)):
    """Computes MRR, Recall@K, and NDCG@5 for a single query."""
    relevance_flags = [
        is_relevant(corpus[idx], idx, target_doc, target_chunk_idx)
        for idx in ranked_indices
    ]

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


# Heuristic: short / jargon-heavy queries benefit from BM25 (alpha=0.3);
# longer, verb-rich natural-language queries benefit from dense (alpha=0.7).
_NL_INDICATORS = {
    "what", "how", "why", "when", "where", "which", "does", "do", "can",
    "will", "should", "could", "would", "explain", "describe", "define",
    "compare", "summarize", "discuss"
}

def _adaptive_alpha(query_text):
    """Returns alpha for _linear_fusion based on query type."""
    tokens = query_text.lower().split()
    if len(tokens) <= 4:
        return 0.3  # short / keyword-like → favour BM25
    nl_hits = sum(1 for t in tokens if t in _NL_INDICATORS)
    return 0.7 if nl_hits >= 1 else 0.3


def _adaptive_hybrid(query_text, b_scores, d_scores):
    alpha = _adaptive_alpha(query_text)
    return _linear_fusion(b_scores, d_scores, alpha)


# ===========================================================
# Benchmark Harness
# ===========================================================
def run_evaluation():
    base_dir = os.path.dirname(__file__)
    corpus_dir = os.path.join(base_dir, "corpus")
    if not os.path.exists(corpus_dir):
        corpus_dir = os.path.join(base_dir, "..", "corpus")
    cache_dir = os.path.join(base_dir, ".cache")
    if not os.path.exists(cache_dir):
        cache_dir = os.path.join(base_dir, "..", ".cache")

    print(f"Loading corpus from: {corpus_dir}")
    corpus, total_pages, pdf_paths = load_structured_corpus(corpus_dir, cache_dir=cache_dir)
    print(f"Loaded {len(pdf_paths)} PDFs | Total Pages: {total_pages} | Chunks: {len(corpus)}\n")

    # ----------------------------------------------------------
    # Sanity-check: verify every ground-truth chunk index still
    # points at a chunk from the expected document.  Fails loudly
    # rather than silently producing meaningless zero metrics.
    # ----------------------------------------------------------
    for item in EVAL_DATASET:
        idx = item.get("target_chunk_idx")
        doc = item["target_doc"]
        if idx is not None:
            if idx >= len(corpus):
                raise AssertionError(
                    f"Ground-truth index {idx} for '{doc}' is out of bounds "
                    f"(corpus has {len(corpus)} chunks). Re-run label_chunks.py."
                )
            if doc.lower() not in corpus[idx][:200].lower():
                raise AssertionError(
                    f"Ground-truth index {idx} does not contain expected doc '{doc}'.\n"
                    f"Chunk preview: {corpus[idx][:120]!r}\n"
                    "The corpus may have been re-chunked. Re-run label_chunks.py."
                )
    print("Ground-truth sanity check passed.\n")

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
    # Neural models (SentenceTransformer & CrossEncoder)
    # ----------------------------------------------------------
    reranker = None
    st_model = None
    corpus_st_vecs = None
    if HAS_NEURAL:
        try:
            print("Loading CrossEncoder ('cross-encoder/ms-marco-MiniLM-L-6-v2')...")
            reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("Loading SentenceTransformer ('all-MiniLM-L6-v2')...")
            st_model = SentenceTransformer("all-MiniLM-L6-v2")

            # Corpus-aware cache key: invalidates when PDFs change or chunking params change.
            st_cache_key = compute_cache_key(corpus_dir, extra_tag="st_minilm")
            st_cache_file = os.path.join(cache_dir, f"st_minilm_{st_cache_key}.npy")
            if os.path.exists(st_cache_file):
                print(f"Loading cached SentenceTransformer embeddings: {st_cache_file}")
                corpus_st_vecs = np.load(st_cache_file)
            else:
                print("Encoding corpus with SentenceTransformer (MiniLM)...")
                corpus_st_vecs = st_model.encode(
                    corpus, batch_size=64, show_progress_bar=False, convert_to_numpy=True
                )
                np.save(st_cache_file, corpus_st_vecs)
                print(f"Cached SentenceTransformer embeddings to {st_cache_file}")
            print()
        except Exception as e:
            print(f"Warning: Failed to load neural models: {e}. Running without neural strategies.\n")
            reranker = None
            st_model = None

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
        "12. Adaptive Hybrid",
    ]
    if reranker is not None and st_model is not None and corpus_st_vecs is not None:
        strategy_names.extend([
            "10. Cross-Encoder Re-rank",
            "11. Sentence-Transformer (MiniLM)",
        ])

    results = {name: [] for name in strategy_names}

    print(f"Evaluating {len(EVAL_DATASET)} queries x {len(strategy_names)} strategies...\n")

    for item in EVAL_DATASET:
        q_text = item["query"]
        target_doc = item["target_doc"]
        target_chunk_idx = item.get("target_chunk_idx")

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

        # --- Base candidate rankings ---
        # Strategy 7: dedup-capped list used only for strategy 7 itself.
        rrf_dedup_candidates = _rrf_dedup(b_scores, d_scores, corpus, k=60)

        # Wider pre-dedup RRF list used as the cross-encoder candidate pool (strategy 10).
        rrf_wide = _rrf(b_scores, d_scores, k=60)  # full sorted RRF order, no dedup cap

        # --- Single dispatch path: build ranked list for every strategy ---
        ranked_by_strategy = {
            "1. Pure BM25 (Sparse)":       np.argsort(b_scores)[::-1].tolist(),
            "2. Pure TF-IDF (Dense)":       np.argsort(d_scores)[::-1].tolist(),
            "3. Linear Hybrid (a=0.3)":     _linear_fusion(b_scores, d_scores, 0.3),
            "4. Linear Hybrid (a=0.5)":     _linear_fusion(b_scores, d_scores, 0.5),
            "5. Linear Hybrid (a=0.7)":     _linear_fusion(b_scores, d_scores, 0.7),
            "6. RRF (k=60)":               _rrf(b_scores, d_scores, k=60),
            "7. RRF + Deduplication":       rrf_dedup_candidates,
            "8. RRF + Dedup + MMR":         _rrf_dedup_mmr(
                                                q_vec, b_scores, d_scores, corpus_tfidf, corpus
                                            ),
            "9. PPMI Semantic + BM25 RRF": ppmi_fused,
            "12. Adaptive Hybrid":          _adaptive_hybrid(q_text, b_scores, d_scores),
        }

        # Neural strategy dispatches
        if reranker is not None:
            # Feed the reranker a wide pre-dedup pool of 50, then dedup the re-ranked list.
            ce_pool = rrf_wide[:50]
            ce_pairs = [(q_text, corpus[idx]) for idx in ce_pool]
            ce_scores = reranker.predict(ce_pairs)
            ce_reranked = [ce_pool[i] for i in np.argsort(ce_scores)[::-1]]
            # Dedup after reranking so the reranker isn't penalised by near-duplicate noise.
            ce_deduped = deduplicate_results(ce_reranked, corpus, threshold=0.65, max_results=10)
            ce_remainder = [idx for idx in rrf_wide if idx not in set(ce_pool)]
            ranked_by_strategy["10. Cross-Encoder Re-rank"] = ce_deduped + ce_remainder

        if st_model is not None and corpus_st_vecs is not None:
            q_st_vec = st_model.encode([q_text], convert_to_numpy=True)
            st_sims = cosine_similarity(q_st_vec, corpus_st_vecs).flatten()
            ranked_by_strategy["11. Sentence-Transformer (MiniLM)"] = np.argsort(st_sims)[::-1].tolist()

        for name in strategy_names:
            ranked = ranked_by_strategy[name]
            metrics = evaluate_ranking(ranked, corpus, target_doc, target_chunk_idx=target_chunk_idx)
            results[name].append(metrics)

    # ----------------------------------------------------------
    # Print Summary Table
    # ----------------------------------------------------------
    print(
        f"\n{'Retrieval Strategy':<36}"
        f"{'MRR':<10}{'Recall@1':<12}{'Recall@3':<12}{'Recall@5':<12}{'NDCG@5':<10}"
    )
    print("=" * 94)

    for name in strategy_names:
        ml = results[name]
        print(
            f"{name:<36}"
            f"{np.mean([m['mrr']      for m in ml]):<10.3f}"
            f"{np.mean([m['recall_1'] for m in ml]):<12.3f}"
            f"{np.mean([m['recall_3'] for m in ml]):<12.3f}"
            f"{np.mean([m['recall_5'] for m in ml]):<12.3f}"
            f"{np.mean([m['ndcg_5']   for m in ml]):<10.3f}"
        )

    print()


if __name__ == "__main__":
    run_evaluation()

