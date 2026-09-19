"""
Evaluation Benchmark Runner for Document Hybrid Search.
Runs the 14-query x 15-strategy benchmark suite from the src package.
Supports comprehensive validation with README-formatted output.
"""

import sys
import argparse

def check_dependencies():
    """Check for common missing dependencies and provide clear error messages."""
    missing_deps = []
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import transformers
    except ImportError:
        missing_deps.append("transformers")
    
    try:
        import sentence_transformers
    except ImportError:
        missing_deps.append("sentence-transformers")
    
    try:
        import networkx
    except ImportError:
        missing_deps.append("networkx")
    
    try:
        import sklearn
    except ImportError:
        missing_deps.append("scikit-learn")
    
    if missing_deps:
        print("Error: Missing required dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nPlease install all dependencies by running:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

# Check dependencies before importing heavy ML libraries
check_dependencies()

from src.common.console import ensure_utf8_streams

# Ensure UTF-8 output on Windows
ensure_utf8_streams()

from src.evaluation.harness import EvaluationHarness


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation benchmarks for Document Hybrid Search with comprehensive validation"
    )
    parser.add_argument(
        "--generation",
        action="store_true",
        help="Run generation layer benchmarks (rate limiting, multi-provider testing)"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live provider API calls for generation benchmarks (consumes quota)"
    )
    parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Run comprehensive validation with README-formatted output (retrieval + generation)"
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="parquet",
        choices=["memory", "parquet", "qdrant"],
        help="Storage backend for evaluation (applies to all evaluation modes)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.comprehensive:
            # Use the new comprehensive validation harness
            print("Running comprehensive validation with README-formatted output...")
            from tests.run_comprehensive_validation import ComprehensiveValidationHarness
            
            harness = ComprehensiveValidationHarness(
                storage_backend=args.storage,
                corpus_dir="corpus",
                output_file="VALIDATION_RESULTS.md",
                live_mode=args.live
            )
            
            harness.run_all()
            return 0
        
        if args.generation:
            # Run generation layer benchmarks
            from tests.run_generation_benchmarks import GenerationBenchmarkHarness
            print("Running generation layer benchmarks with real corpus queries...")
            harness = GenerationBenchmarkHarness(live_mode=args.live, storage_backend=args.storage)
            harness.run_all()
        else:
            # Run standard retrieval evaluation (default behavior)
            print("Running retrieval evaluation with 14 ground-truth queries from real corpus...")
            harness = EvaluationHarness(storage_backend=args.storage)
            harness.run()
        
        return 0
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
        return 130
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
