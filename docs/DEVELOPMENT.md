# Development Guide

## Environment Setup
```bash
# Clone and install
git clone <repo-url>
cd <repo-name>
pip install -e .[test,dev]
pre-commit install
```

## Development Workflow
• **Local Testing**: `pytest -n auto -v`
• **Code Quality**: `ruff check . && bandit -r src`
• **Coverage**: `pytest --cov=src --cov-report=term-missing`

## Architecture
• Multi-agent pipeline for email triage
• See [ADR docs](adr/) for architecture decisions
• Review [ARCHITECTURE.md](../ARCHITECTURE.md) for system design

## Documentation
• API docs: Auto-generated from docstrings
• Architecture decisions: [`docs/adr/`](adr/)
• Project status: [`docs/status/`](status/)

## Debugging
• Use `logging_utils.py` for structured logging
• Enable debug mode: `LOGLEVEL=DEBUG`
• Metrics available via `metrics_export.py`

## References
• [Python Testing Guide](https://docs.python.org/3/library/unittest.html)
• [CrewAI Documentation](https://docs.crewai.com/)
• [Email Processing Best Practices](https://realpython.com/python-send-email/)