"""
Hybrid Document Search using Modern Scientific Libraries:
- numpy: Vectorized array math, score fusion, and argsort ranking
- scipy: Sparse matrix representations
- scikit-learn: TF-IDF vectorization, cosine similarity, and min-max scaling
- rank_bm25 / numpy: High-performance BM25Okapi sparse keyword retrieval
- sentence-transformers (optional): Neural dense embeddings when available

This script serves as the library-accelerated counterpart to hybrid_search_from_scratch.py.
"""

import os
import sys

from corpus_loader import extract_text_from_pdf, load_pdf_corpus  # noqa: F401 (re-exported)
import argparse
import numpy as np

# Reconfigure stdout for UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale

# Optional BM25 library
try:
    from rank_bm25 import BM25Okapi
    HAS_RANK_BM25 = True
except ImportError:
    HAS_RANK_BM25 = False

# Optional Sentence-Transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


# ===========================================================
# Vectorized BM25 Fallback (Pure NumPy)
# ===========================================================
class NumpyBM25:
    """Vectorized BM25 using NumPy arrays when rank-bm25 is not installed."""
    def __init__(self, tokenized_corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.N = len(tokenized_corpus)
        self.doc_lens = np.array([len(doc) for doc in tokenized_corpus], dtype=np.float32)
        self.avgdl = float(np.mean(self.doc_lens)) if self.N > 0 else 1.0

        # Term to doc counts
        self.doc_freqs = []
        df_counts = {}
        for doc in tokenized_corpus:
            counts = {}
            for term in doc:
                counts[term] = counts.get(term, 0) + 1
            self.doc_freqs.append(counts)
            for term in set(doc):
                df_counts[term] = df_counts.get(term, 0) + 1

        # Precompute IDF with Robertson-Spärck Jones formula
        self.idf = {
            t: np.log((self.N - freq + 0.5) / (freq + 0.5) + 1.0)
            for t, freq in df_counts.items()
        }

    def get_scores(self, query_tokens):
        scores = np.zeros(self.N, dtype=np.float32)
        for term in query_tokens:
            if term not in self.idf:
                continue
            idf_val = self.idf[term]
            # Gather frequencies across documents
            freqs = np.array([doc.get(term, 0) for doc in self.doc_freqs], dtype=np.float32)
            mask = freqs > 0
            if not np.any(mask):
                continue
            denom = freqs[mask] + self.k1 * (1.0 - self.b + self.b * (self.doc_lens[mask] / self.avgdl))
            scores[mask] += idf_val * (freqs[mask] * (self.k1 + 1.0) / denom)
        return scores


# ===========================================================
# Normalization Helper (NumPy)
# ===========================================================
def normalize_scores(scores):
    """Min-max normalizes scores to [0.0, 1.0] using scikit-learn or numpy."""
    s_min, s_max = np.min(scores), np.max(scores)
    if s_max == s_min:
        return np.zeros_like(scores, dtype=np.float32)
    return (scores - s_min) / (s_max - s_min)


# ===========================================================
# Main Search Pipeline
# ===========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Search using NumPy, Scipy, Scikit-Learn, and Rank-BM25."
    )
    parser.add_argument(
        "query",
        nargs="*",
        default=["Binding", "Constraint", "Thesis", "in", "LLM", "agent", "execution", "harness"],
        help="Search query string",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Weight for dense semantic score vs sparse BM25 (default: 0.6)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results to return (default: 5)",
    )
    parser.add_argument(
        "--neural",
        action="store_true",
        help="Use sentence-transformers neural embeddings instead of TF-IDF vectors",
    )
    args = parser.parse_args()

    active_query = " ".join(args.query)

    # 1. Discover and chunk PDF corpus
    corpus_dir = os.path.join(os.path.dirname(__file__), "corpus")
    if not os.path.isdir(corpus_dir):
        corpus_dir = os.path.join(os.path.dirname(__file__), "..", "corpus")
    if os.path.isdir(corpus_dir):
        pdf_paths = [
            os.path.join(corpus_dir, f)
            for f in sorted(os.listdir(corpus_dir))
            if f.lower().endswith(".pdf")
        ]
    else:
        fallback = os.path.join(os.path.dirname(__file__), "2605.23950v1.pdf")
        pdf_paths = [fallback] if os.path.exists(fallback) else []

    if not pdf_paths:
        print("Error: No PDF documents found in corpus/ or workspace.")
        sys.exit(1)

    print(f"Loaded {len(pdf_paths)} PDF(s) from: {corpus_dir}")
    corpus, total_pages = load_pdf_corpus(pdf_paths)
    print(f"Total Pages: {total_pages} | Chunks in Corpus: {len(corpus)}")
    print(f"Query: {active_query!r}")
    print(f"Configuration: Alpha={args.alpha:.2f} (Dense), 1-Alpha={1.0-args.alpha:.2f} (BM25)\n")

    # 2. Sparse Search: BM25 (via rank_bm25 or NumPy)
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    tokenized_query = active_query.lower().split()

    if HAS_RANK_BM25:
        bm25_model = BM25Okapi(tokenized_corpus)
        bm25_raw = np.array(bm25_model.get_scores(tokenized_query), dtype=np.float32)
    else:
        bm25_model = NumpyBM25(tokenized_corpus)
        bm25_raw = bm25_model.get_scores(tokenized_query)

    # 3. Dense Search: Scikit-Learn TF-IDF or Neural Sentence-Transformers
    if args.neural and HAS_SENTENCE_TRANSFORMERS:
        print("Using Sentence-Transformers ('all-MiniLM-L6-v2') for dense neural embeddings...")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        corpus_embeddings = embedder.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)
        query_embedding = embedder.encode([active_query], convert_to_numpy=True, normalize_embeddings=True)
        cosine_raw = cosine_similarity(query_embedding, corpus_embeddings).flatten()
    else:
        if args.neural and not HAS_SENTENCE_TRANSFORMERS:
            print("Note: sentence-transformers not installed. Falling back to Scikit-Learn TF-IDF.")
        # Scikit-Learn TfidfVectorizer (sparse matrix representation via SciPy)
        vectorizer = TfidfVectorizer(
            lowercase=True,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b"
        )
        corpus_tfidf = vectorizer.fit_transform(corpus)
        query_tfidf = vectorizer.transform([active_query])
        cosine_raw = cosine_similarity(query_tfidf, corpus_tfidf).flatten()

    # 4. Normalization (NumPy / Scikit-Learn)
    bm25_norm = normalize_scores(bm25_raw)
    cosine_norm = normalize_scores(cosine_raw)

    # 5. Vectorized Hybrid Fusion (NumPy)
    hybrid_scores = (args.alpha * cosine_norm) + ((1.0 - args.alpha) * bm25_norm)

    # 6. NumPy Fast Ranking
    ranked_indices = np.argsort(hybrid_scores)[::-1]
    top_k = min(args.top_k, len(corpus))

    # 7. Print Results Table
    dense_label = "Neural" if (args.neural and HAS_SENTENCE_TRANSFORMERS) else "Cosine"
    print(f"{'Rank':<5}{'Hybrid':<10}{dense_label:<10}{'BM25':<10}Chunk Excerpt")
    print("-" * 110)
    for rank, idx in enumerate(ranked_indices[:top_k], start=1):
        raw_text = corpus[idx].replace("\n", " ")
        snippet = raw_text[:88] + "..." if len(raw_text) > 88 else raw_text
        print(
            f"{rank:<5}{hybrid_scores[idx]:<10.3f}"
            f"{cosine_norm[idx]:<10.3f}{bm25_norm[idx]:<10.3f}{snippet}"
        )

    # 8. Display Rank #1 Match
    best_idx = ranked_indices[0]
    print("\n" + "=" * 80)
    print(f"TOP MATCH (Rank #1) [Hybrid Score: {hybrid_scores[best_idx]:.3f}]:")
    print("=" * 80)
    print(corpus[best_idx])


if __name__ == "__main__":
    main()
