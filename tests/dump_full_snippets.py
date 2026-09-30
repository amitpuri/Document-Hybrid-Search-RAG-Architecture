import os
import sys

sys.path.insert(0, os.getcwd())  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.engine import HybridSearchEngine  # noqa: E402

engine = HybridSearchEngine.from_corpus("corpus", storage_backend="parquet")
strategies = [
    "bm25",
    "tfidf",
    "linear_0.3",
    "linear_0.5",
    "linear_0.7",
    "rrf",
    "rrf_dedup",
    "rrf_dedup_mmr",
    "ppmi",
    "cross_encoder",
    "sentence_transformer",
    "adaptive",
    "specter2",
    "rrf_graph_dedup_mmr",
]
queries = [
    (
        "a",
        "How does the Binding Constraint Thesis affect harness comparisons?",
    ),
    ("b", "Dynamic Tiered AgentRunner Framework Risk Adaptive Tiering"),
]

out_path = os.path.join(os.path.dirname(__file__), "full_snippets.txt")
with open(out_path, "w", encoding="utf-8") as f:
    for q_label, q_text in queries:
        f.write(f"==================== QUERY {q_label}: {q_text} ====================\n\n")
        for strat in strategies:
            res = engine.search(q_text, strategy=strat, top_k=3)
            f.write(f"#### Strategy: `{strat}`\n")
            top1 = res[0]
            msg = (
                f"- **Top-1 Source:** `{top1.chunk.doc_name}` | "
                f"Page {top1.chunk.page_num} | § {top1.chunk.section} "
                f"(Chunk ID: {top1.chunk.chunk_id}, Score: {top1.score:.3f})\n"
            )
            f.write(msg)
            clean_text = top1.chunk.text.strip().replace("\n", "\n> ")
            f.write(f"- **Complete Snippet:**\n> {clean_text}\n\n")

print("DUMP COMPLETE")
