# Contributing to CrewAI Email Triage

## Quick Start
• Fork repository from `main` branch
• Install: `pip install -e .[test]` + `pre-commit install`
• Test: `pytest -n auto -q` 
• Lint: `ruff check . && bandit -r src -q`

## Guidelines
• Follow [Conventional Commits](https://conventionalcommits.org/) format
• Reference [GitHub Flow](https://guides.github.com/introduction/flow/) for branching
• See [Python PEP 8](https://pep8.org/) for style guidelines
• Include tests and documentation with changes

## Pull Requests
• Link to relevant issues
• Ensure CI passes
• Follow [PR best practices](https://github.com/blog/1943-how-to-write-the-perfect-pull-request)

## Questions?
• Check existing [issues](../../issues) and [discussions](../../discussions)
• Review [Code of Conduct](CODE_OF_CONDUCT.md)
• See [Security Policy](SECURITY.md) for vulnerability reports
