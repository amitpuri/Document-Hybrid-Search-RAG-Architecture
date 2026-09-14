"""
Distributional Semantic Hybrid Search From Scratch Using Pointwise Mutual Information (PPMI).

Zero External Dependencies (Pure Python Standard Library: math, collections).

Why this matters:
Standard bag-of-words cosine similarity is just another exact keyword matcher.
If a query uses "oversight" and a chunk says "governability", BoW gives 0.0 similarity.

This module implements:
1. Term-term sliding window co-occurrence matrix from the corpus.
2. Positive Pointwise Mutual Information (PPMI):
     PPMI(w1, w2) = max(0, log2( P(w1, w2) / (P(w1) * P(w2)) ))
3. Distributional word embeddings: Top-50 strongest PPMI context dimensions per word.
4. Passage semantic embeddings: Mean pooling of word vectors.
5. True distributional cosine similarity without external libraries.
6. Fusion with BM25 via Reciprocal Rank Fusion (RRF).
"""

import os
import sys
import math
from collections import Counter, defaultdict

# Ensure UTF-8 output encoding for console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from corpus_loader import load_structured_corpus



# ===========================================================
# Standard Tokenizer & Stopwords
# ===========================================================
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
    "by", "of", "from", "as", "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "this", "that", "these", "those",
    "it", "its", "we", "our", "you", "your", "they", "their", "he", "she", "which",
    "can", "will", "would", "should", "could", "all", "any", "both", "each", "more"
}

def tokenize(text):
    cleaned = "".join(ch if ch.isalnum() else " " for ch in text.lower())
    return cleaned.split()


# ===========================================================
# Distributional Semantics: PPMI Word Embeddings from Scratch
# ===========================================================
class PPMIEmbeddings:
    """
    Constructs term-term PPMI vectors from corpus co-occurrences.
    Keeps top-50 strongest context associations per word for clean, noise-filtered embeddings.
    """
    def __init__(self, tokenized_corpus, window_size=5, vocab_size=1500, max_context_per_word=50):
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.max_context_per_word = max_context_per_word

        # 1. Select top content vocabulary
        term_freqs = Counter()
        for doc in tokenized_corpus:
            content_tokens = [t for t in doc if t not in STOPWORDS and len(t) > 2]
            term_freqs.update(content_tokens)

        top_terms = [term for term, _ in term_freqs.most_common(vocab_size)]
        self.vocab = set(top_terms)

        # 2. Count co-occurrences within sliding window
        cooccur = defaultdict(Counter)
        unigram_counts = Counter()
        total_cooccur = 0

        for doc in tokenized_corpus:
            filtered = [t for t in doc if t in self.vocab]
            n = len(filtered)
            for i, target in enumerate(filtered):
                unigram_counts[target] += 1
                start = max(0, i - window_size)
                end = min(n, i + window_size + 1)
                for j in range(start, end):
                    if i != j:
                        ctx = filtered[j]
                        cooccur[target][ctx] += 1
                        total_cooccur += 1

        self.total_cooccur = max(1, total_cooccur)
        self.unigram_counts = unigram_counts

        # 3. Build sparse PPMI vectors: keep top strongest associations
        self.word_vectors = {}
        for word, ctx_counts in cooccur.items():
            p_w = unigram_counts[word]
            if p_w == 0:
                continue

            word_ppmis = []
            for ctx, count in ctx_counts.items():
                p_c = unigram_counts[ctx]
                # Expected co-occurrence count under independence
                expected = (p_w * p_c * 2 * self.window_size) / self.total_cooccur
                if expected > 0:
                    pmi = math.log2((count * self.total_cooccur) / (p_w * p_c * 2 * self.window_size))
                    if pmi > 0:  # Positive PMI
                        word_ppmis.append((ctx, pmi))

            # Keep top-K strongest context signals
            word_ppmis.sort(key=lambda x: x[1], reverse=True)
            self.word_vectors[word] = dict(word_ppmis[:self.max_context_per_word])

    def embed_text(self, tokens):
        """Mean pools PPMI word vectors into a passage embedding vector."""
        combined = Counter()
        count = 0
        for token in tokens:
            if token in self.word_vectors:
                for ctx, weight in self.word_vectors[token].items():
                    combined[ctx] += weight
                count += 1

        if count == 0:
            return {}, 0.0

        avg_vec = {ctx: val / count for ctx, val in combined.items()}
        mag = math.sqrt(sum(v * v for v in avg_vec.values()))
        return avg_vec, mag


# ===========================================================
# Sparse Vector Cosine Similarity
# ===========================================================
def sparse_cosine_similarity(vec_a, mag_a, vec_b, mag_b):
    """Computes cosine similarity between two sparse dictionary vectors."""
    if mag_a == 0.0 or mag_b == 0.0 or not vec_a or not vec_b:
        return 0.0

    # Dot product over intersecting keys
    smaller, larger = (vec_a, vec_b) if len(vec_a) < len(vec_b) else (vec_b, vec_a)
    dot = sum(val * larger[k] for k, val in smaller.items() if k in larger)
    if dot <= 0:
        return 0.0

    return dot / (mag_a * mag_b)


# ===========================================================
# BM25 From Scratch
# ===========================================================
class BM25:
    def __init__(self, corpus_tokens, k1=1.5, b=0.75):
        self.corpus_tokens = corpus_tokens
        self.k1 = k1
        self.b = b
        self.N = len(corpus_tokens)
        self.doc_lengths = [len(doc) for doc in corpus_tokens]
        self.avgdl = sum(self.doc_lengths) / max(1, self.N)
        self.doc_freqs = [Counter(doc) for doc in corpus_tokens]
        self.idf = self._compute_idf()

    def _compute_idf(self):
        df = Counter()
        for doc in self.corpus_tokens:
            for term in set(doc):
                df[term] += 1
        return {
            term: math.log((self.N - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }

    def get_scores(self, query_tokens):
        scores = [0.0] * self.N
        for i in range(self.N):
            doc_len = self.doc_lengths[i]
            freqs = self.doc_freqs[i]
            score = 0.0
            for term in query_tokens:
                if term in freqs:
                    f = freqs[term]
                    idf = self.idf.get(term, 0.0)
                    denom = f + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                    score += idf * (f * (self.k1 + 1.0) / denom)
            scores[i] = score
        return scores


# ===========================================================
# Reciprocal Rank Fusion (RRF)
# ===========================================================
def rrf_fusion(ranking_a, ranking_b, k=60):
    scores = {}
    for rank, idx in enumerate(ranking_a):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, idx in enumerate(ranking_b):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True), scores


# ===========================================================
# Main Search Demonstration
# ===========================================================
def main():
    query_text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Binding Constraint Thesis in LLM agent execution harness"
    corpus_dir = os.path.join(os.path.dirname(__file__), "corpus")
    if not os.path.exists(corpus_dir):
        corpus_dir = os.path.join(os.path.dirname(__file__), "..", "corpus")
    cache_dir = os.path.join(os.path.dirname(__file__), ".cache")
    if not os.path.exists(cache_dir):
        cache_dir = os.path.join(os.path.dirname(__file__), "..", ".cache")

    print(f"Loading corpus from: {corpus_dir}")
    corpus, total_pages, pdf_paths = load_structured_corpus(corpus_dir, cache_dir=cache_dir)
    print(f"Loaded {len(pdf_paths)} PDFs | Total Pages: {total_pages} | Chunks: {len(corpus)}")
    print(f"Query: {query_text!r}\n")

    print("Building PPMI distributional embeddings from scratch (window=5, vocab=1500, top-50 context)...")
    tokenized_corpus = [tokenize(doc) for doc in corpus]
    query_tokens = tokenize(query_text)

    # 1. Build PPMI Embeddings
    embedder = PPMIEmbeddings(tokenized_corpus, window_size=5, vocab_size=1500, max_context_per_word=50)
    print(f"Trained vocabulary of {len(embedder.vocab)} content words with distributional context vectors.")

    # 2. Embed passages and query
    chunk_embeddings = [embedder.embed_text(tokens) for tokens in tokenized_corpus]
    query_vec, query_mag = embedder.embed_text(query_tokens)

    # 3. Dense Semantic Cosine Similarity (PPMI Distributional)
    semantic_scores = [
        sparse_cosine_similarity(query_vec, query_mag, c_vec, c_mag)
        for c_vec, c_mag in chunk_embeddings
    ]
    semantic_ranking = sorted(range(len(corpus)), key=lambda i: semantic_scores[i], reverse=True)

    # 4. Sparse BM25
    bm25 = BM25(tokenized_corpus)
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_ranking = sorted(range(len(corpus)), key=lambda i: bm25_scores[i], reverse=True)

    # 5. Reciprocal Rank Fusion (RRF)
    fused_ranking, rrf_scores = rrf_fusion(semantic_ranking, bm25_ranking, k=60)

    # 6. Display Results
    print("\n" + f"{'Rank':<5}{'RRF':<10}{'PPMI-Sim':<10}{'BM25':<10}Chunk Excerpt")
    print("-" * 110)
    for rank, idx in enumerate(fused_ranking[:5], start=1):
        raw_text = corpus[idx].replace("\n", " ")
        snippet = raw_text[:88] + "..." if len(raw_text) > 88 else raw_text
        print(
            f"{rank:<5}{rrf_scores[idx]:<10.4f}"
            f"{semantic_scores[idx]:<10.3f}{bm25_scores[idx]:<10.3f}{snippet}"
        )

    best_idx = fused_ranking[0]
    print("\n" + "=" * 80)
    print(f"TOP MATCH (Rank #1) [RRF Score: {rrf_scores[best_idx]:.4f} | PPMI Semantic: {semantic_scores[best_idx]:.3f}]:")
    print("=" * 80)
    print(corpus[best_idx])


if __name__ == "__main__":
    main()
