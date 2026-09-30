#!/bin/bash
set -e

echo "=== Running CI checks locally ==="

# Lint job (use system Python for linting)
echo "=== Lint & Format ==="
python -m pip install black flake8
python -m black --check src tests --target-version py311
python -m flake8 src tests --max-line-length=100 --extend-ignore=E203

# Test job (use venv for testing)
echo "=== Tests ==="
./venv/Scripts/python.exe -m pip install -e ".[dev]"
./venv/Scripts/python.exe -m pytest tests/ -v --cov=src --cov-report=xml

# Security job
echo "=== Security Check ==="
python -m pip install bandit
python -m bandit -r src --exit-zero -f json -o bandit-report.json

# Docs job
echo "=== Documentation ==="
test -f README.md && echo "README.md exists" || (echo "ERROR: README.md not found" && exit 1)
test -f docs/roadmap.md && echo "docs/roadmap.md exists" || (echo "ERROR: docs/roadmap.md not found" && exit 1)

echo "=== All CI checks passed ==="