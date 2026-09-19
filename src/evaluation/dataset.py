"""
Curated Ground-Truth Benchmark Dataset and Sanity Validation.
"""

# TODO (P3 — Evaluation Rigor):
# All 14 benchmark entries below specify an explicit `target_chunk_idx`,
# which means the substring fallback in metrics.py is_relevant() is
# currently unreachable.
# Future work:
#   1. Expand to 50+ queries to make per-query MRR variance statistically
#      meaningful (currently 1 query flip = 1/14 ≈ 0.071 MRR shift,
#      exceeding several strategy gaps).
#   2. Split into a dev set (used for tuning DEFAULT_RRF_K,
#      DEFAULT_DEDUP_THRESHOLD, DEFAULT_MMR_LAMBDA in config.py) and a
#      held-out test set (for reported numbers) to avoid implicit
#      hyperparameter overfitting to the benchmark.

from typing import Any, Dict, List, Sequence

EVAL_DATASET: List[Dict[str, Any]] = [
    {
        "query": "Binding Constraint Thesis in LLM agent execution harness",
        "target_doc": "2605.23950v1.pdf",
        "target_chunk_idx": 1561,  # § 3 The Binding Constraint Thesis
        "target_entities": [
            "Binding Constraint Thesis",
            "harness",
            "execution",
        ],
        "target_relations": [("Binding Constraint Thesis", "harness", "execution")],
        "is_multihop": False,
    },
    {
        "query": ("Terminal Agents Suffice for Enterprise Automation " "StarShell command line"),
        "target_doc": "2604.00073v3.pdf",
        "target_chunk_idx": 564,  # § 3.2 StarShell: Terminal-Based Agent
        "target_entities": ["StarShell", "Terminal", "Enterprise Automation"],
        "target_relations": [("StarShell", "Terminal", "Automation")],
        "is_multihop": False,
    },
    {
        "query": "Dynamic Tiered AgentRunner Framework Risk Adaptive Tiering",
        "target_doc": "2605.10223v1.pdf",
        "target_chunk_idx": 865,  # § Abstract / Framework intro
        "target_entities": [
            "AgentRunner",
            "Risk Adaptive Tiering",
            "Framework",
        ],
        "target_relations": [("AgentRunner", "Tiering", "Risk")],
        "is_multihop": False,
    },
    {
        "query": ("Position AI Evaluations Should be Grounded on a " "Theory of Capability"),
        "target_doc": "2509.19590v2.pdf",
        "target_chunk_idx": 396,  # § Abstract / Position definition
        "target_entities": ["AI Evaluations", "Theory of Capability"],
        "target_relations": [("Evaluations", "Theory", "Capability")],
        "is_multihop": False,
    },
    {
        "query": "Economy of AI agents market forces and firm sizes",
        "target_doc": "2509.01063v1.pdf",
        "target_chunk_idx": 321,  # § 3 / 3.1 Firm sizes
        "target_entities": ["AI agents", "market forces", "firm sizes"],
        "target_relations": [("market forces", "shape", "firm sizes")],
        "is_multihop": True,
    },
    {
        "query": ("Partially Observed Markov Decision Processes " "belief state filtering"),
        "target_doc": "1604.08127v1.pdf",
        "target_chunk_idx": 0,  # § Overview / POMDP intro
        "target_entities": ["POMDP", "belief state", "filtering"],
        "target_relations": [("POMDP", "belief state", "filtering")],
        "is_multihop": True,
    },
    {
        "query": ("Thinking with Looped Flows recurrent reasoning depth"),
        "target_doc": "2609.11801v1.pdf",
        "target_chunk_idx": 1963,  # § 1 Contributions / Looped flows
        "target_entities": ["Looped Flows", "recurrent reasoning", "depth"],
        "target_relations": [("Looped Flows", "recurrent", "reasoning")],
        "is_multihop": False,
    },
    {
        "query": ("Can coding agents be general agents " "web browser interaction"),
        "target_doc": "2604.13107v1.pdf",
        "target_chunk_idx": 820,  # § 1 Title & Abstract
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
        "query": ("Your model already knows hidden representation " "extraction"),
        "target_doc": "2609.11310v1.pdf",
        "target_chunk_idx": 1725,  # § 1 Title & Abstract
        "target_entities": ["hidden representation", "extraction"],
        "target_relations": [("representation", "hidden", "extraction")],
        "is_multihop": False,
    },
    # --- NL-phrased queries: stress-test α=0.7 branch ---
    {
        "query": (
            "How does the Binding Constraint Thesis affect " "harness comparisons across models"
        ),
        "target_doc": "2605.23950v1.pdf",
        "target_chunk_idx": 1561,  # same target as query 1
        "target_entities": ["Binding Constraint Thesis", "harness", "models"],
        "target_relations": [("Binding Constraint Thesis", "affect", "harness")],
        "is_multihop": True,
    },
    {
        "query": ("What risk-tiering mechanisms does the " "AgentRunner framework apply"),
        "target_doc": "2605.10223v1.pdf",
        "target_chunk_idx": 865,  # same target as query 3
        "target_entities": ["risk-tiering", "AgentRunner", "framework"],
        "target_relations": [("AgentRunner", "tiering", "risk")],
        "is_multihop": False,
    },
    {
        "query": ("How do POMDP belief states update after " "receiving new observations"),
        "target_doc": "1604.08127v1.pdf",
        "target_chunk_idx": 0,  # same target as query 6
        "target_entities": ["POMDP", "belief states", "observations"],
        "target_relations": [("belief states", "update", "observations")],
        "is_multihop": True,
    },
    {
        "query": ("What market forces shape the organization " "and size of AI agent firms"),
        "target_doc": "2509.01063v1.pdf",
        "target_chunk_idx": 321,  # same target as query 5
        "target_entities": ["market forces", "organization", "firms"],
        "target_relations": [("market forces", "shape", "firms")],
        "is_multihop": True,
    },
    # --- Expanded Corpus Benchmark Queries ---
    {
        "query": ("AlphaGenome regulatory variant effect " "prediction non-coding DNA"),
        "target_doc": "s41586-025-10014-0.pdf",
        "target_chunk_idx": 6056,  # § Overview / AlphaGenome intro
        "target_entities": ["AlphaGenome", "regulatory variant", "prediction"],
        "target_relations": [("AlphaGenome", "variant", "prediction")],
        "is_multihop": False,
    },
    {
        "query": ("Scalable watermarking for identifying large " "language model outputs SynthID"),
        "target_doc": "s41586-024-08025-4.pdf",
        "target_chunk_idx": 5941,  # § Overview / Scalable watermarking
        "target_entities": ["watermarking", "SynthID", "outputs"],
        "target_relations": [("watermarking", "identifying", "outputs")],
        "is_multihop": False,
    },
    {
        "query": ("Procedural Graphs Self-Evolving Execution " "Structures for LLM Agents"),
        "target_doc": "2609.09153v1.pdf",
        "target_chunk_idx": 5222,  # § 1. Introduction / Procedural Graphs
        "target_entities": [
            "Procedural Graphs",
            "Execution Structures",
            "LLM Agents",
        ],
        "target_relations": [("Procedural Graphs", "Execution Structures", "Agents")],
        "is_multihop": False,
    },
    {
        "query": ("Analyzing and Predicting Token Consumption " "in Agentic Coding Tasks"),
        "target_doc": "2604.22750v2.pdf",
        "target_chunk_idx": 4492,  # § Abstract / Token Consumption
        "target_entities": ["Token Consumption", "Agentic Coding Tasks"],
        "target_relations": [("Token Consumption", "Coding", "Tasks")],
        "is_multihop": False,
    },
    {
        "query": ("CTIFOUNDRY AGENT-NATIVE CORPUS SCAFFOLD " "FOR CYBER THREAT INTELLIGENCE"),
        "target_doc": "2608.18613v1.pdf",
        "target_chunk_idx": 4896,  # § 1 INTRODUCTION / CTIFoundry
        "target_entities": [
            "CTIFOUNDRY",
            "AGENT-NATIVE",
            "CYBER THREAT INTELLIGENCE",
        ],
        "target_relations": [("CTIFOUNDRY", "SCAFFOLD", "THREAT")],
        "is_multihop": False,
    },
    {
        "query": ("CliniCARE-Bench Clinical Calibrated Audit " "of Medical Reasoning in EHR"),
        "target_doc": "2608.07796v1.pdf",
        "target_chunk_idx": 4621,  # § Abstract / CliniCARE-Bench
        "target_entities": ["CliniCARE-Bench", "Medical Reasoning", "EHR"],
        "target_relations": [("CliniCARE-Bench", "Medical Reasoning", "EHR")],
        "is_multihop": False,
    },
    {
        "query": ("Thinking Fast Slow and Artificial Tri-System " "Theory Cognitive Surrender"),
        "target_doc": "ssrn-6097646.pdf",
        "target_chunk_idx": 6291,  # § Abstract / Tri-System Theory
        "target_entities": [
            "Tri-System Theory",
            "Cognitive Surrender",
            "Reasoning",
        ],
        "target_relations": [("Tri-System Theory", "Cognitive Surrender", "Reasoning")],
        "is_multihop": False,
    },
    {
        "query": ("AI Safety Not Optional autonomous agent " "scaffolds and software harness"),
        "target_doc": "2609.10630v1.pdf",
        "target_chunk_idx": 5429,  # § Introduction / AI Safety
        "target_entities": ["AI Safety", "scaffolds", "harness"],
        "target_relations": [("AI Safety", "scaffolds", "harness")],
        "is_multihop": False,
    },
]


def validate_ground_truth(dataset: List[Dict[str, Any]], corpus_texts: Sequence[str]) -> None:
    """
    Sanity checks that every ground-truth chunk index points to a chunk
    from the expected document. Fails loudly with AssertionError on drift.

    NOTE ON VALIDATION GUARANTEES:
    - This function acts as an index-drift safeguard: when re-chunking
      parameters (such as max_words or overlap_sentences) are modified,
      absolute chunk indices shift. This check validates that
      target_chunk_idx still resides within the target document.
    - Semantic entity and relation validation (target_entities,
      target_relations) is evaluated downstream during ranking
      evaluation via evaluate_ranking() in `src.evaluation.metrics`
      using deterministic regex word boundary matching across the
      top-5 retrieved context.
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
                    f"Ground-truth index {idx} does not contain expected "
                    f"doc '{doc}'.\n"
                    f"Chunk preview: {corpus_texts[idx][:120]!r}\n"
                    "Corpus chunking may have drifted. Re-run dataset "
                    "calibration if chunking parameters changed."
                )
