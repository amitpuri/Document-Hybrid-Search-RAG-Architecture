"""
Core Logic of BM25 (sparse/keyword search) and Cosine Similarity (dense/semantic search)
Implemented from scratch using ONLY Python's standard library (math, collections).

No numpy, no sklearn, no sentence-transformers.
"embeddings" here are simple word-frequency vectors we build ourselves,
so you can see exactly how the math works under the hood.
"""

import os
import sys
import math
from collections import Counter

# Ensure UTF-8 output encoding for console (e.g. Windows cp1252 handling of math/unicode symbols)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from corpus_loader import extract_text_from_pdf, load_pdf_corpus  # noqa: F401 (re-exported)


# ===========================================================
# Corpus & Default Query
# ===========================================================
CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")
if not os.path.isdir(CORPUS_DIR):
    CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "corpus")

# Load array of PDF paths from corpus/ directory
if os.path.isdir(CORPUS_DIR):
    PDF_PATHS = [
        os.path.join(CORPUS_DIR, f)
        for f in sorted(os.listdir(CORPUS_DIR))
        if f.lower().endswith(".pdf")
    ]
else:
    # Fallback to single PDF if corpus folder is absent
    single_pdf = os.path.join(os.path.dirname(__file__), "2605.23950v1.pdf")
    PDF_PATHS = [single_pdf] if os.path.exists(single_pdf) else []

corpus, total_pages = load_pdf_corpus(PDF_PATHS)
query = "Binding Constraint Thesis in LLM agent execution harness"


# ===========================================================
# Helper: basic tokenizer (lowercase + strip punctuation)
# ===========================================================
def tokenize(text):
    cleaned = "".join(ch if ch.isalnum() else " " for ch in text.lower())
    return cleaned.split()


# ===========================================================
# 1. BM25 FROM SCRATCH
# ===========================================================
class BM25:
    def __init__(self, corpus_tokens, k1=1.5, b=0.75):
        self.corpus_tokens = corpus_tokens
        self.k1 = k1
        self.b = b
        self.N = len(corpus_tokens)
        self.doc_lengths = [len(doc) for doc in corpus_tokens]
        self.avgdl = sum(self.doc_lengths) / self.N
        self.doc_freqs = [Counter(doc) for doc in corpus_tokens]  # term counts per doc
        self.idf = self._compute_idf()

    def _compute_idf(self):
        # count how many documents each term appears in
        df = Counter()
        for doc in self.corpus_tokens:
            for term in set(doc):
                df[term] += 1

        idf = {}
        for term, freq in df.items():
            # standard BM25 idf formula (with +1 smoothing to avoid negatives)
            idf[term] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1)
        return idf

    def score(self, query_tokens, doc_index):
        score = 0.0
        doc_len = self.doc_lengths[doc_index]
        freqs = self.doc_freqs[doc_index]

        for term in query_tokens:
            if term not in freqs:
                continue
            f = freqs[term]
            idf = self.idf.get(term, 0)
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * (numerator / denominator)
        return score

    def get_scores(self, query_tokens):
        return [self.score(query_tokens, i) for i in range(self.N)]


# ===========================================================
# 2. "EMBEDDINGS" + COSINE SIMILARITY FROM SCRATCH
#    (using simple word-frequency vectors as a stand-in for
#    real dense embeddings — same vector math applies)
# ===========================================================
def build_vocab(all_token_lists):
    vocab = set()
    for tokens in all_token_lists:
        vocab.update(tokens)
    return sorted(vocab)  # fixed order needed to align vector dimensions


def vectorize(tokens, vocab, vocab_index=None):
    if vocab_index is None:
        vocab_index = {term: i for i, term in enumerate(vocab)}
    vec = [0] * len(vocab)
    for term, count in Counter(tokens).items():
        if term in vocab_index:
            vec[vocab_index[term]] = count
    return vec


def dot_product(v1, v2):
    # Sparse-aware dot product: if v1 is sparse, only sum across its non-zero entries
    non_zero_indices = [i for i, x in enumerate(v1) if x != 0]
    if len(non_zero_indices) < len(v1) // 10:
        return sum(v1[i] * v2[i] for i in non_zero_indices)
    return sum(a * b for a, b in zip(v1, v2))


def magnitude(v):
    return math.sqrt(sum(a * a for a in v))


def cosine_similarity(v1, v2, mag1=None, mag2=None):
    denom = (mag1 if mag1 is not None else magnitude(v1)) * (mag2 if mag2 is not None else magnitude(v2))
    if denom == 0:
        return 0.0
    return dot_product(v1, v2) / denom


# ===========================================================
# 3. NORMALIZATION (min-max, so BM25 and cosine become comparable)
# ===========================================================
def min_max_normalize(scores):
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


# ===========================================================
# RUN EVERYTHING
# ===========================================================
if __name__ == "__main__":
    # Support custom query from command line if provided
    active_query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else query

    if len(PDF_PATHS) > 1:
        print(f"Loaded {len(PDF_PATHS)} PDFs from: {CORPUS_DIR}")
    else:
        print(f"Loaded PDF: {PDF_PATHS[0] if PDF_PATHS else 'None'}")
    print(f"Total Pages: {total_pages} | Chunks in Corpus: {len(corpus)}")
    print(f"Query: {active_query!r}\n")

    # --- tokenize corpus + query ---
    corpus_tokens = [tokenize(doc) for doc in corpus]
    query_tokens = tokenize(active_query)

    # --- BM25 (sparse) ---
    bm25 = BM25(corpus_tokens)
    bm25_scores = bm25.get_scores(query_tokens)

    # --- Cosine similarity (dense, via word-frequency vectors) ---
    vocab = build_vocab(corpus_tokens + [query_tokens])
    vocab_index = {term: i for i, term in enumerate(vocab)}
    doc_vectors = [vectorize(tokens, vocab, vocab_index) for tokens in corpus_tokens]
    query_vector = vectorize(query_tokens, vocab, vocab_index)
    query_mag = magnitude(query_vector)
    cosine_scores = [cosine_similarity(query_vector, dv, mag1=query_mag) for dv in doc_vectors]

    # --- Normalize both score sets to [0, 1] ---
    bm25_norm = min_max_normalize(bm25_scores)
    cosine_norm = min_max_normalize(cosine_scores)

    # --- Weighted hybrid fusion ---
    alpha = 0.6  # weight given to dense/cosine score
    hybrid_scores = [
        alpha * c + (1 - alpha) * b for c, b in zip(cosine_norm, bm25_norm)
    ]

    # --- Rank by hybrid score ---
    ranking = sorted(range(len(corpus)), key=lambda i: hybrid_scores[i], reverse=True)

    # --- Display Top Results ---
    top_k = min(5, len(corpus))
    print(f"{'Rank':<5}{'Hybrid':<10}{'Cosine':<10}{'BM25':<10}Chunk Excerpt")
    print("-" * 110)
    for rank, idx in enumerate(ranking[:top_k], start=1):
        raw_text = corpus[idx].replace("\n", " ")
        snippet = raw_text[:90] + "..." if len(raw_text) > 90 else raw_text
        print(
            f"{rank:<5}{hybrid_scores[idx]:<10.3f}"
            f"{cosine_norm[idx]:<10.3f}{bm25_norm[idx]:<10.3f}{snippet}"
        )

    # --- Print Top-1 Best Match in Full ---
    best_idx = ranking[0]
    print("\n" + "=" * 80)
    print(f"TOP MATCH (Rank #1) [Hybrid Score: {hybrid_scores[best_idx]:.3f}]:")
    print("=" * 80)
    print(corpus[best_idx])

