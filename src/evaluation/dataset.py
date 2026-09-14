"""
Curated Ground-Truth Benchmark Dataset and Sanity Validation.
"""

from typing import List, Dict, Any, Sequence

EVAL_DATASET: List[Dict[str, Any]] = [
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
    {
        "query": "How does the Binding Constraint Thesis affect harness comparisons across models",
        "target_doc": "2605.23950v1.pdf",
        "target_chunk_idx": 1561,  # same target as query 1, NL phrasing -> α=0.7
    },
    {
        "query": "What risk-tiering mechanisms does the AgentRunner framework apply",
        "target_doc": "2605.10223v1.pdf",
        "target_chunk_idx": 865,   # same target as query 3, NL phrasing -> α=0.7
    },
    {
        "query": "How do POMDP belief states update after receiving new observations",
        "target_doc": "1604.08127v1.pdf",
        "target_chunk_idx": 0,     # same target as query 6, NL phrasing -> α=0.7
    },
    {
        "query": "What market forces shape the organization and size of AI agent firms",
        "target_doc": "2509.01063v1.pdf",
        "target_chunk_idx": 321,   # same target as query 5, NL phrasing -> α=0.7
    },
]


def validate_ground_truth(dataset: List[Dict[str, Any]], corpus_texts: Sequence[str]) -> None:
    """
    Sanity checks that every ground-truth chunk index points to a chunk
    from the expected document. Fails loudly with an AssertionError if drift occurs.
    """
    corpus_size = len(corpus_texts)
    for item in dataset:
        idx = item.get("target_chunk_idx")
        doc = item["target_doc"]
        if idx is not None:
            if idx >= corpus_size:
                raise AssertionError(
                    f"Ground-truth index {idx} for '{doc}' is out of bounds "
                    f"(corpus has {corpus_size} chunks)."
                )
            if doc.lower() not in corpus_texts[idx][:200].lower():
                raise AssertionError(
                    f"Ground-truth index {idx} does not contain expected doc '{doc}'.\n"
                    f"Chunk preview: {corpus_texts[idx][:120]!r}\n"
                    "Corpus chunking may have drifted."
                )
