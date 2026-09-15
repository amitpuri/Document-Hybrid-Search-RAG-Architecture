import subprocess
import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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
]

queries = [
    ("a", "How does the Binding Constraint Thesis affect harness comparisons?"),
    ("b", "Dynamic Tiered AgentRunner Framework Risk Adaptive Tiering"),
]

for q_label, query in queries:
    print(f"==================== QUERY {q_label}: {query} ====================")
    for strat in strategies:
        cmd = [
            sys.executable,
            "-m",
            "src.cli",
            "search",
            query,
            "--strategy",
            strat,
            "--top-k",
            "3",
            "--corpus",
            "corpus",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        print(f"--- STRATEGY: {strat} ---")
        if result.returncode != 0:
            print(f"ERROR: {result.stderr}")
        else:
            # Filter out tqdm / loading weights lines if present, keep search output
            lines = result.stdout.splitlines()
            search_lines = []
            capture = False
            for l in lines:
                if l.startswith("Executing search:") or l.startswith("Rank"):
                    capture = True
                if capture:
                    search_lines.append(l)
            if search_lines:
                print("\n".join(search_lines))
            else:
                print(result.stdout)
        print()
