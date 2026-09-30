#!/bin/bash

# Load environment variables from .env
sh load_env.sh

# Create results directory
mkdir -p results/minimal

# Define all strategies to test
STRATEGIES=(
    "bm25"
    "tfidf"
    "linear_0.3"
    "linear_0.5"
    "linear_0.7"
    "rrf"
    "rrf_dedup"
    "rrf_dedup_mmr"
    "ppmi"
    "cross_encoder"
    "sentence_transformer"
    "adaptive"
    "specter2"
    "rrf_graph_dedup_mmr"
    "qdrant"
)

echo "Running minimal validation for all strategies with live LLM providers..."
echo "Environment loaded from .env"
echo ""

for strategy in "${STRATEGIES[@]}"; do
    echo "========================================"
    echo "Testing strategy: $strategy"
    echo "========================================"
    
    # Run minimal validation with fast placeholder data
    python tests/test_minimal_validation.py \
        "$strategy" \
        "results/minimal/MINIMAL_VALIDATION-${strategy}.md"
    
    if [ $? -eq 0 ]; then
        echo "✓ $strategy completed successfully"
    else
        echo "✗ $strategy failed"
    fi
    echo ""
done

echo "========================================"
echo "All minimal validations completed"
echo "========================================"
echo "Results saved to results/minimal/"

