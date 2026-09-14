"""
Evaluation Benchmark Runner for Document Hybrid Search.
Runs the 14-query x 12-strategy benchmark suite from the src package.
"""

import sys

# Reconfigure stdout for UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.evaluation.harness import EvaluationHarness

if __name__ == "__main__":
    harness = EvaluationHarness()
    harness.run()
