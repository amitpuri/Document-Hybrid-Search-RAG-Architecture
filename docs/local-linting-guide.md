# Local Linting & Type Checking Guide

This guide provides commands to run locally before pushing code, matching the CI pipeline checks.

## Prerequisites

Install development dependencies:
```bash
pip install -e ".[dev]"
```

## Type Checking (mypy)

### Run mypy with same settings as CI:
```bash
mypy src --ignore-missing-imports --warn-unused-ignores --no-incremental
```

### Fix issues step by step:
1. Run mypy to get a list of errors
2. Address **actual type errors** (not just unused ignores)
3. Remove **unused `type: ignore` comments** - these cause CI to fail
4. Verify the error code matches what mypy reports (e.g., `[misc]`, `[assignment]`, `[no-any-return]`)

### Common mypy errors:

| Error | Code | Fix |
|-------|------|-----|
| `Unused "type: ignore" comment` | `[unused-ignore]` | Remove the comment entirely |
| `Cannot assign to a type` | `[misc]` | Use `# type: ignore[misc]` |
| `Incompatible types in assignment` | `[assignment]` | Use `# type: ignore[assignment]` |
| Functions with no return type annotation | `[annotation-unchecked]` | Either add return type or use `--check-untyped-defs` |

## Code Formatting (black)

### Check formatting without making changes:
```bash
black --check src tests
```

### Auto-format code:
```bash
black src tests
```

## Linting (flake8)

### Run flake8 with CI settings:
```bash
flake8 src tests --max-line-length=100 --extend-ignore=E203
```

Common issues:
- `E501`: Line too long - break into multiple lines or use Black
- `F401`: Unused import - remove it
- `E203`: Whitespace before `:` - often a Black formatting conflict

## Security Scanning (bandit)

### Run security checks:
```bash
bandit -r src --exit-zero -f json -o bandit-report.json
```

Review the JSON report for security issues.

## Run All Checks (Full CI Pipeline Locally)

```bash
# 1. Format with Black
black src tests

# 2. Check with flake8
flake8 src tests --max-line-length=100 --extend-ignore=E203

# 3. Type check with mypy
mypy src --ignore-missing-imports --warn-unused-ignores --no-incremental

# 4. Run tests
pytest tests/ -v --cov=src --cov-report=html

# 5. Security scan
bandit -r src --exit-zero -f json -o bandit-report.json
```

## Recommended Workflow

Before committing:

```bash
# Format code
black src tests

# Run all checks
mypy src --ignore-missing-imports --warn-unused-ignores --no-incremental
flake8 src tests --max-line-length=100 --extend-ignore=E203
pytest tests/ -v

# Then commit and push
git add .
git commit -m "your message"
git push origin main
```

## IDE Integration

### VSCode

Install extensions:
- `ms-python.python` - Python extension
- `ms-python.vscode-pylance` - Type checking

Add to `.vscode/settings.json`:
```json
{
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "python.linting.flake8Args": ["--max-line-length=100", "--extend-ignore=E203"],
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length=100"],
  "[python]": {
    "editor.defaultFormatter": "ms-python.python",
    "editor.formatOnSave": true
  }
}
```

### PyCharm

1. Go to Settings → Tools → Python Integrated Tools
2. Set Default Test Runner to `pytest`
3. Set Default Linter to `flake8`
4. Configure Black formatter: Settings → Tools → Python → Black
5. Enable "Reformat code" on save

## Troubleshooting

### mypy errors persist after removing `type: ignore`

Run with `--no-incremental` to clear cached analysis:
```bash
mypy src --ignore-missing-imports --warn-unused-ignores --no-incremental
```

### Type checking takes too long

Use `--incremental` (default) for faster subsequent runs. Only use `--no-incremental` for CI.

### Different Python versions have different type stub issues

The project supports Python 3.8+. Mypy is configured for Python 3.11. If you hit version-specific stub issues (like numpy 2.5+ requiring Python 3.12+), constrain the dependency version in `pyproject.toml`.
