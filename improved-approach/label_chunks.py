"""
Discovery helper: Run once to inspect top candidate chunks per query
and determine the definitive ground-truth target_chunk_idx.
"""

import os
import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

from corpus_loader import load_structured_corpus
from hybrid_search_rrf import reciprocal_rank_fusion, deduplicate_results
from eval_harness import EVAL_DATASET

def main():
    base_dir = os.path.dirname(__file__)
    corpus_dir = os.path.join(base_dir, "corpus")
    if not os.path.exists(corpus_dir):
        corpus_dir = os.path.join(base_dir, "..", "corpus")
    cache_dir = os.path.join(base_dir, ".cache")
    if not os.path.exists(cache_dir):
        cache_dir = os.path.join(base_dir, "..", ".cache")

    corpus, total_pages, pdf_paths = load_structured_corpus(corpus_dir, cache_dir=cache_dir)
    print(f"Loaded {len(pdf_paths)} PDFs | Total Pages: {total_pages} | Chunks: {len(corpus)}\n")

    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    vectorizer = TfidfVectorizer(lowercase=True, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
    corpus_tfidf = vectorizer.fit_transform(corpus)

    print("=" * 80)
    print("CHUNK GROUND TRUTH DISCOVERY HELPER")
    print("=" * 80)

    for i, item in enumerate(EVAL_DATASET, 1):
        q_text = item["query"]
        target_doc = item["target_doc"]
        print(f"\n[{i}/10] Query: \"{q_text}\"")
        print(f"Target Doc: {target_doc}")
        print("-" * 80)

        q_tokens = q_text.lower().split()
        b_scores = np.array(bm25.get_scores(q_tokens), dtype=np.float32)

        q_vec = vectorizer.transform([q_text])
        d_scores = cosine_similarity(q_vec, corpus_tfidf).flatten().astype(np.float32)

        b_rank = np.argsort(b_scores)[::-1]
        d_rank = np.argsort(d_scores)[::-1]
        fused_idx, _ = reciprocal_rank_fusion(b_rank, d_rank, k=60)
        top_indices = deduplicate_results(fused_idx, corpus, threshold=0.65, max_results=5)

        for rank, idx in enumerate(top_indices, 1):
            chunk = corpus[idx]
            is_target = target_doc.lower() in chunk[:120].lower()
            target_flag = "[TARGET DOC MATCH]" if is_target else "[OTHER DOC]"
            print(f"  Rank #{rank} | Chunk #{idx} {target_flag}")
            print(f"    Preview: {repr(chunk[:220])}")
            print()

if __name__ == "__main__":
    main()
