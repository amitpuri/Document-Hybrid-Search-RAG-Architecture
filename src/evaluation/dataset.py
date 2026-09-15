"""
Curated Ground-Truth Benchmark Dataset and Sanity Validation.
"""

from typing import List, Dict, Any, Sequence

EVAL_DATASET: List[Dict[str, Any]] = [
    {
        "query": "Binding Constraint Thesis in LLM agent execution harness",
        "target_doc": "2605.23950v1.pdf",
        "target_chunk_idx": 1561,  # § 3 The Binding Constraint Thesis
        "target_entities": ["Binding Constraint Thesis", "harness", "execution"],
        "target_relations": [("Binding Constraint Thesis", "harness", "execution")],
        "is_multihop": False,
    },
    {
        "query": "Terminal Agents Suffice for Enterprise Automation StarShell command line",
        "target_doc": "2604.00073v3.pdf",
        "target_chunk_idx": 564,   # § 3.2 StarShell: A Terminal-Based Enterprise Agent
        "target_entities": ["StarShell", "Terminal", "Enterprise Automation"],
        "target_relations": [("StarShell", "Terminal", "Automation")],
        "is_multihop": False,
    },
    {
        "query": "Dynamic Tiered AgentRunner Framework Risk Adaptive Tiering",
        "target_doc": "2605.10223v1.pdf",
        "target_chunk_idx": 865,   # § Abstract / Framework intro
        "target_entities": ["AgentRunner", "Risk Adaptive Tiering", "Framework"],
        "target_relations": [("AgentRunner", "Tiering", "Risk")],
        "is_multihop": False,
    },
    {
        "query": "Position AI Evaluations Should be Grounded on a Theory of Capability",
        "target_doc": "2509.19590v2.pdf",
        "target_chunk_idx": 396,   # § Abstract / Position definition
        "target_entities": ["AI Evaluations", "Theory of Capability"],
        "target_relations": [("Evaluations", "Theory", "Capability")],
        "is_multihop": False,
    },
    {
        "query": "Economy of AI agents market forces and firm sizes",
        "target_doc": "2509.01063v1.pdf",
        "target_chunk_idx": 321,   # § 3 Organizations of AI agents / 3.1 Firm sizes
        "target_entities": ["AI agents", "market forces", "firm sizes"],
        "target_relations": [("market forces", "shape", "firm sizes")],
        "is_multihop": True,
    },
    {
        "query": "Partially Observed Markov Decision Processes belief state filtering",
        "target_doc": "1604.08127v1.pdf",
        "target_chunk_idx": 0,     # § Overview / POMDP intro
        "target_entities": ["POMDP", "belief state", "filtering"],
        "target_relations": [("POMDP", "belief state", "filtering")],
        "is_multihop": True,
    },
    {
        "query": "Thinking with Looped Flows recurrent reasoning depth",
        "target_doc": "2609.11801v1.pdf",
        "target_chunk_idx": 1963,  # § 1 Contributions / Looped flows framework
        "target_entities": ["Looped Flows", "recurrent reasoning", "depth"],
        "target_relations": [("Looped Flows", "recurrent", "reasoning")],
        "is_multihop": False,
    },
    {
        "query": "Can coding agents be general agents web browser interaction",
        "target_doc": "2604.13107v1.pdf",
        "target_chunk_idx": 820,   # § 1 Title & Abstract
        "target_entities": ["coding agents", "web browser", "general agents"],
        "target_relations": [("coding agents", "browser", "interaction")],
        "is_multihop": True,
    },
    {
        "query": "Code as Agent Harness context window constraint",
        "target_doc": "2605.18747v1.pdf",
        "target_chunk_idx": 1149,  # § 4.1 Context Window Constraint
        "target_entities": ["Agent Harness", "context window", "constraint"],
        "target_relations": [("Harness", "context window", "constraint")],
        "is_multihop": False,
    },
    {
        "query": "Your model already knows hidden representation extraction",
        "target_doc": "2609.11310v1.pdf",
        "target_chunk_idx": 1725,  # § 1 Title & Abstract
        "target_entities": ["hidden representation", "extraction"],
        "target_relations": [("representation", "hidden", "extraction")],
        "is_multihop": False,
    },
    # --- NL-phrased queries: designed to stress-test Strategy 12's α=0.7 branch ---
    {
        "query": "How does the Binding Constraint Thesis affect harness comparisons across models",
        "target_doc": "2605.23950v1.pdf",
        "target_chunk_idx": 1561,  # same target as query 1, NL phrasing -> α=0.7
        "target_entities": ["Binding Constraint Thesis", "harness", "models"],
        "target_relations": [("Binding Constraint Thesis", "affect", "harness")],
        "is_multihop": True,
    },
    {
        "query": "What risk-tiering mechanisms does the AgentRunner framework apply",
        "target_doc": "2605.10223v1.pdf",
        "target_chunk_idx": 865,   # same target as query 3, NL phrasing -> α=0.7
        "target_entities": ["risk-tiering", "AgentRunner", "framework"],
        "target_relations": [("AgentRunner", "tiering", "risk")],
        "is_multihop": False,
    },
    {
        "query": "How do POMDP belief states update after receiving new observations",
        "target_doc": "1604.08127v1.pdf",
        "target_chunk_idx": 0,     # same target as query 6, NL phrasing -> α=0.7
        "target_entities": ["POMDP", "belief states", "observations"],
        "target_relations": [("belief states", "update", "observations")],
        "is_multihop": True,
    },
    {
        "query": "What market forces shape the organization and size of AI agent firms",
        "target_doc": "2509.01063v1.pdf",
        "target_chunk_idx": 321,   # same target as query 5, NL phrasing -> α=0.7
        "target_entities": ["market forces", "organization", "firms"],
        "target_relations": [("market forces", "shape", "firms")],
        "is_multihop": True,
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
